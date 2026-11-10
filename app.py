import base64
import csv
import io
import os
import time
import uuid
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from functools import wraps

import boto3
import numpy as np
import pytz
from decouple import config
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from PIL import Image, ImageOps
from sqlalchemy import event, func, or_
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import NullPool
from werkzeug.security import check_password_hash, generate_password_hash
import sqlite3

face_recognition = None
def _get_face_recognition():
    global face_recognition
    if face_recognition is None:
        import face_recognition as fr
        face_recognition = fr
    return face_recognition

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IST = pytz.timezone('Asia/Kolkata')

# Vercel sets VERCEL=1. Its filesystem is read-only except for /tmp, and the
# request body limit is 4.5MB, so paths and limits have to change there.
IS_VERCEL = bool(os.environ.get('VERCEL') or os.environ.get('VERCEL_ENV'))

# Static files live in public/static so Vercel's CDN can serve them. Flask
# serves the same folder locally under /static.
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, 'public', 'static'),
    static_url_path='/static',
)
app.config['SECRET_KEY'] = config('SECRET_KEY', default='attendance-purple-secret-change-me')
app.config['UPLOAD_FOLDER'] = (
    '/tmp/uploads' if IS_VERCEL else os.path.join(BASE_DIR, 'instance', 'uploads')
)
app.config['MAX_CONTENT_LENGTH'] = (4 if IS_VERCEL else 16) * 1024 * 1024
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
except OSError:
    # Read-only filesystem (Vercel without a writable /tmp path).
    app.logger.warning('Upload folder could not be created: %s', app.config['UPLOAD_FOLDER'])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def now_ist():
    return datetime.now(IST)


def today_date():
    return now_ist().date()


def parse_date(value, default):
    try:
        return datetime.strptime(value, '%Y-%m-%d').date()
    except (TypeError, ValueError):
        return default


def inr(value):
    if value is None:
        return '—'
    return '₹{:,.0f}'.format(float(value))


app.jinja_env.filters['inr'] = inr


# ---------------------------------------------------------------------------
# Database (Neon PostgreSQL, with local SQLite fallback for development)
# ---------------------------------------------------------------------------
def _normalize_db_url(url):
    """Turn Neon/Heroku style URLs into a SQLAlchemy psycopg3 URL."""
    url = (url or '').strip()
    if url.startswith('postgres://'):
        url = 'postgresql://' + url[len('postgres://'):]
    if url.startswith('postgresql://'):
        url = 'postgresql+psycopg://' + url[len('postgresql://'):]
    return url


DATABASE_URL = config('DATABASE_URL', default='').strip()
USING_SQLITE_FALLBACK = not bool(DATABASE_URL)
if USING_SQLITE_FALLBACK:
    sqlite_path = '/tmp/attendance_local.db' if IS_VERCEL else os.path.join(BASE_DIR, 'attendance_local.db')
    DATABASE_URL = 'sqlite:///' + sqlite_path
else:
    DATABASE_URL = _normalize_db_url(DATABASE_URL)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
if DATABASE_URL.startswith('sqlite'):
    sqlite_options = {'connect_args': {'check_same_thread': False}}
    if IS_VERCEL:
        # /tmp is ephemeral; do not keep connections between invocations.
        sqlite_options['poolclass'] = NullPool
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = sqlite_options
elif IS_VERCEL:
    # Serverless invocations should not reuse connections across instances.
    # Neon's pooled connection string handles pooling on the database side.
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'poolclass': NullPool,
        'pool_pre_ping': True,
        'connect_args': {'connect_timeout': 10},
    }
else:
    # Neon can drop idle connections; pre-ping + recycle keeps the pool healthy.
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }

db = SQLAlchemy(app)


@event.listens_for(Engine, 'connect')
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    """SQLite ignores foreign keys unless the pragma is enabled per connection."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute('PRAGMA foreign_keys=ON')
        cursor.close()


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(255), default='', nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='employee', nullable=False)
    employee_id = db.Column(
        db.Integer,
        db.ForeignKey('employees.id', ondelete='CASCADE'),
        nullable=True,
        unique=True,
    )
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: now_ist())

    employee = db.relationship('Employee', back_populates='user')

    @property
    def is_admin(self):
        return self.role == 'admin'

    @property
    def initials(self):
        return (self.username or '?')[:2].upper()

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Employee(db.Model):
    __tablename__ = 'employees'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(160), nullable=False)
    employee_code = db.Column(db.String(80), unique=True, nullable=False, index=True)
    phone = db.Column(db.String(40), default='', nullable=False)
    email = db.Column(db.String(255), default='', nullable=False)
    monthly_salary = db.Column(db.Numeric(12, 2), nullable=True)
    photo_key = db.Column(db.String(255), nullable=True)
    photo_storage = db.Column(db.String(20), default='local')
    face_encoding = db.Column(db.JSON, nullable=True)
    active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: now_ist())

    user = db.relationship(
        'User',
        back_populates='employee',
        uselist=False,
        cascade='all, delete-orphan',
        single_parent=True,
    )
    attendances = db.relationship(
        'Attendance',
        back_populates='employee',
        cascade='all, delete-orphan',
        order_by='Attendance.date.desc()',
    )
    leave_requests = db.relationship(
        'LeaveRequest', back_populates='employee', cascade='all, delete-orphan'
    )
    notice_requests = db.relationship(
        'NoticeRequest', back_populates='employee', cascade='all, delete-orphan'
    )

    @property
    def photo_url(self):
        return photo_url_for(self)


class Attendance(db.Model):
    __tablename__ = 'attendance'
    __table_args__ = (
        db.UniqueConstraint('employee_id', 'date', name='uq_attendance_employee_date'),
    )

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(
        db.Integer,
        db.ForeignKey('employees.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )
    date = db.Column(db.Date, nullable=False, index=True)
    time = db.Column(db.String(8), default='')  # check-in time
    timestamp = db.Column(db.DateTime(timezone=True), default=lambda: now_ist())
    out_time = db.Column(db.String(8), default='')
    out_timestamp = db.Column(db.DateTime(timezone=True), nullable=True)
    out_method = db.Column(db.String(20), default='')
    status = db.Column(db.String(20), default='Present', nullable=False)
    method = db.Column(db.String(20), default='manual', nullable=False)
    check_in_lat = db.Column(db.String(40), default='')
    check_in_lng = db.Column(db.String(40), default='')
    location_id = db.Column(db.Integer, db.ForeignKey('office_locations.id', ondelete='SET NULL'), nullable=True)

    employee = db.relationship('Employee', back_populates='attendances')
    location = db.relationship('OfficeLocation', back_populates='attendances')

    @property
    def check_in_time(self):
        return self.time or ''

    @property
    def check_out_time(self):
        return self.out_time or ''


# ---------------------------------------------------------------------------
# Office locations (geofencing)
# ---------------------------------------------------------------------------
class OfficeLocation(db.Model):
    __tablename__ = 'office_locations'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    radius_meters = db.Column(db.Integer, default=100, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: now_ist())

    attendances = db.relationship('Attendance', back_populates='location')

    @property
    def display(self):
        return f"{self.name} ({self.radius_meters}m)"

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'radius_meters': self.radius_meters,
            'is_active': self.is_active,
        }


# ---------------------------------------------------------------------------
# Leave requests
# ---------------------------------------------------------------------------
class LeaveRequest(db.Model):
    __tablename__ = 'leave_requests'

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(
        db.Integer, db.ForeignKey('employees.id', ondelete='CASCADE'), nullable=False, index=True
    )
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    reason = db.Column(db.Text, default='')
    status = db.Column(db.String(20), default='pending', nullable=False)  # pending|approved|rejected
    admin_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    admin_note = db.Column(db.String(255), default='')
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: now_ist())
    reviewed_at = db.Column(db.DateTime(timezone=True), nullable=True)

    employee = db.relationship('Employee', back_populates='leave_requests')
    admin = db.relationship('User')

    @property
    def days(self):
        return (self.end_date - self.start_date).days + 1

    @property
    def status_badge(self):
        return {
            'pending': ('amber', 'Pending'),
            'approved': ('success', 'Approved'),
            'rejected': ('danger', 'Rejected'),
        }.get(self.status, ('amber', 'Pending'))


# ---------------------------------------------------------------------------
# Notice period requests
# ---------------------------------------------------------------------------
class NoticeRequest(db.Model):
    __tablename__ = 'notice_requests'

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(
        db.Integer, db.ForeignKey('employees.id', ondelete='CASCADE'), nullable=False, index=True
    )
    notice_start = db.Column(db.Date, nullable=False)
    notice_end = db.Column(db.Date, nullable=False)
    reason = db.Column(db.Text, default='')
    status = db.Column(db.String(20), default='pending', nullable=False)  # pending|approved|rejected
    admin_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    admin_note = db.Column(db.String(255), default='')
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: now_ist())
    reviewed_at = db.Column(db.DateTime(timezone=True), nullable=True)

    employee = db.relationship('Employee', back_populates='notice_requests')
    admin = db.relationship('User')

    @property
    def days(self):
        return (self.notice_end - self.notice_start).days + 1

    @property
    def status_badge(self):
        return {
            'pending': ('amber', 'Pending'),
            'approved': ('success', 'Approved'),
            'rejected': ('danger', 'Rejected'),
        }.get(self.status, ('amber', 'Pending'))


# ---------------------------------------------------------------------------
# Cloudflare R2 (S3 compatible). Falls back to local disk when not configured.
# ---------------------------------------------------------------------------
R2_ACCOUNT_ID = config('R2_ACCOUNT_ID', default='').strip()
R2_ACCESS_KEY_ID = config('R2_ACCESS_KEY_ID', default='').strip()
R2_SECRET_ACCESS_KEY = config('R2_SECRET_ACCESS_KEY', default='').strip()
R2_BUCKET = config('R2_BUCKET', default='').strip()
R2_PUBLIC_URL = config('R2_PUBLIC_URL', default='').strip().rstrip('/')
R2_ENDPOINT = config('R2_ENDPOINT', default='').strip()
if not R2_ENDPOINT and R2_ACCOUNT_ID:
    R2_ENDPOINT = f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com'
R2_ENABLED = bool(R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_BUCKET)

_r2_client = None


def r2_client():
    global _r2_client
    if _r2_client is None:
        _r2_client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name='auto',
        )
    return _r2_client


def save_photo(raw, content_type='image/jpeg'):
    """Store a photo in R2 when configured, otherwise on local disk."""
    key = f"employees/{uuid.uuid4().hex}.jpg"
    if R2_ENABLED:
        r2_client().put_object(
            Bucket=R2_BUCKET,
            Key=key,
            Body=raw,
            ContentType=content_type or 'image/jpeg',
        )
        return key, 'r2'

    local_name = os.path.basename(key)
    with open(os.path.join(app.config['UPLOAD_FOLDER'], local_name), 'wb') as fh:
        fh.write(raw)
    return local_name, 'local'


def _store_photo(raw, content_type):
    try:
        return save_photo(raw, content_type)
    except Exception as exc:
        app.logger.error('Photo upload failed: %s', exc)
        flash('Photo upload fail hui. R2 settings aur network check karein.', 'error')
        return None, None


def delete_photo_key(key, storage):
    if not key:
        return
    if storage == 'r2':
        if R2_ENABLED:
            try:
                r2_client().delete_object(Bucket=R2_BUCKET, Key=key)
            except Exception:
                pass
    else:
        path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(key))
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def delete_photo(employee):
    delete_photo_key(employee.photo_key, employee.photo_storage)


def photo_url_for(employee):
    key = employee.photo_key
    if not key:
        return None
    if employee.photo_storage == 'r2':
        if R2_PUBLIC_URL:
            return f"{R2_PUBLIC_URL}/{key}"
        if not R2_ENABLED:
            return None
        try:
            return r2_client().generate_presigned_url(
                'get_object',
                Params={'Bucket': R2_BUCKET, 'Key': key},
                ExpiresIn=3600,
            )
        except Exception:
            return None
    return url_for('uploaded_file', filename=key)


def read_face_encoding(raw):
    """Return the first face encoding from raw image bytes, or None."""
    fr = _get_face_recognition()
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
    image_np = np.array(image)
    encodings = fr.face_encodings(image_np)
    return encodings[0] if encodings else None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please login to continue.'
login_manager.login_message_category = 'info'
login_manager.session_protection = 'strong'

# Remember-me cookie 30 days tak valid rahega taaki baar-baar login na karna pade
app.config['REMEMBER_COOKIE_DURATION'] = 30 * 24 * 60 * 60  # 30 days
app.config['REMEMBER_COOKIE_SECURE'] = False
app.config['REMEMBER_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 30 * 24 * 60 * 60  # 30 days


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if not current_user.is_admin:
            flash('Sirf admin is page ko access kar sakta hai.', 'error')
            return redirect(url_for('dashboard'))
        return view(*args, **kwargs)

    return wrapped


def employee_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if current_user.is_admin:
            flash('Admin apni attendance mark nahi kar sakta.', 'error')
            return redirect(url_for('dashboard'))
        return view(*args, **kwargs)

    return wrapped


@app.context_processor
def inject_db_state():
    return {
        'using_sqlite_fallback': USING_SQLITE_FALLBACK,
        'is_vercel': IS_VERCEL,
        'r2_enabled': R2_ENABLED,
        'ephemeral_uploads': IS_VERCEL and not R2_ENABLED,
        'max_upload_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
    }


# ---------------------------------------------------------------------------
# Schema + default admin
# ---------------------------------------------------------------------------
def ensure_schema_migrations():
    """Apply lightweight column migrations for existing databases."""
    try:
        cols = db.session.execute(db.text("SELECT column_name FROM information_schema.columns WHERE table_name='attendance'"))
        names = {r[0] for r in cols}
        if 'out_time' not in names:
            db.session.execute(db.text("ALTER TABLE attendance ADD COLUMN out_time VARCHAR(8) DEFAULT ''"))
        if 'out_timestamp' not in names:
            db.session.execute(db.text("ALTER TABLE attendance ADD COLUMN out_timestamp TIMESTAMPTZ"))
        if 'out_method' not in names:
            db.session.execute(db.text("ALTER TABLE attendance ADD COLUMN out_method VARCHAR(20) DEFAULT ''"))
        if 'check_in_lat' not in names:
            db.session.execute(db.text("ALTER TABLE attendance ADD COLUMN check_in_lat VARCHAR(40) DEFAULT ''"))
        if 'check_in_lng' not in names:
            db.session.execute(db.text("ALTER TABLE attendance ADD COLUMN check_in_lng VARCHAR(40) DEFAULT ''"))
        if 'location_id' not in names:
            db.session.execute(db.text("ALTER TABLE attendance ADD COLUMN location_id INTEGER REFERENCES office_locations(id) ON DELETE SET NULL"))
        db.session.commit()
    except Exception:
        db.session.rollback()
        try:
            cols = db.session.execute(db.text("PRAGMA table_info(attendance)"))
            names = {r[1] for r in cols}
            if 'out_time' not in names:
                db.session.execute(db.text("ALTER TABLE attendance ADD COLUMN out_time VARCHAR(8) DEFAULT ''"))
            if 'out_timestamp' not in names:
                db.session.execute(db.text("ALTER TABLE attendance ADD COLUMN out_timestamp DATETIME"))
            if 'out_method' not in names:
                db.session.execute(db.text("ALTER TABLE attendance ADD COLUMN out_method VARCHAR(20) DEFAULT ''"))
            if 'check_in_lat' not in names:
                db.session.execute(db.text("ALTER TABLE attendance ADD COLUMN check_in_lat VARCHAR(40) DEFAULT ''"))
            if 'check_in_lng' not in names:
                db.session.execute(db.text("ALTER TABLE attendance ADD COLUMN check_in_lng VARCHAR(40) DEFAULT ''"))
            if 'location_id' not in names:
                db.session.execute(db.text("ALTER TABLE attendance ADD COLUMN location_id INTEGER"))
            db.session.commit()
        except Exception:
            db.session.rollback()


def _table_exists(table_name):
    try:
        if db.engine.dialect.name == 'sqlite':
            rows = db.session.execute(db.text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"), {'t': table_name}).fetchall()
        else:
            rows = db.session.execute(db.text("SELECT table_name FROM information_schema.tables WHERE table_name=:t"), {'t': table_name}).fetchall()
        return bool(rows)
    except Exception:
        return False


def _ensure_new_tables():
    """Create new tables for geofencing, leaves and notice periods if missing."""
    try:
        db.create_all()
        db.session.commit()
    except Exception:
        db.session.rollback()


def ensure_default_admin():
    """Create the first admin account if the database has none."""
    if db.session.execute(db.select(User).where(User.role == 'admin')).first():
        return
    username = config('ADMIN_USERNAME', default='admin')
    if db.session.execute(db.select(User).where(User.username == username)).first():
        return
    admin = User(
        username=username,
        email=config('ADMIN_EMAIL', default='admin@example.com'),
        role='admin',
    )
    admin.set_password(config('ADMIN_PASSWORD', default='admin123'))
    db.session.add(admin)
    db.session.commit()


def init_db():
    with app.app_context():
        # Two cold starts can race on CREATE TABLE with Neon. Retry briefly so
        # the loser of the race sees the tables created by the winner.
        last_error = None
        for attempt in range(3):
            try:
                db.create_all()
                ensure_schema_migrations()
                ensure_default_admin()
                return
            except Exception as exc:  # pragma: no cover - startup diagnostics
                last_error = exc
                db.session.rollback()
                if attempt < 2:
                    time.sleep(1)
        if last_error is not None:
            raise last_error


try:
    init_db()
except Exception as exc:  # pragma: no cover - startup diagnostics
    raise RuntimeError(
        'Database connect/init fail hui. DATABASE_URL check karein '
        '(.env ya Vercel environment variables; Neon connection string, '
        'sslmode=require ke saath). Original error: '
        f'{exc}'
    ) from exc


@app.errorhandler(413)
def upload_too_large(_error):
    limit_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    flash(f'File bahut badi hai. Maximum {limit_mb}MB allowed hai.', 'error')
    target = request.referrer
    if not target or not target.startswith(request.host_url):
        target = url_for('dashboard')
    return redirect(target, code=303)


# ---------------------------------------------------------------------------
# Employee / attendance helpers
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_employee(employee_id):
    try:
        return db.session.get(Employee, int(employee_id))
    except (TypeError, ValueError):
        return None


def parse_salary(raw):
    """Return (Decimal|None, error_message|None). Blank is allowed."""
    raw = (raw or '').strip()
    if not raw:
        return None, None
    try:
        value = Decimal(raw).quantize(Decimal('0.01'))
        if value <= 0:
            raise InvalidOperation
    except (InvalidOperation, ValueError):
        return None, 'Salary ek positive number honi chahiye (ya khaali chhodein).'
    return value, None


def present_employee_ids(date):
    rows = db.session.execute(
        db.select(Attendance.employee_id).where(
            Attendance.date == date,
            Attendance.status == 'Present',
        )
    ).scalars()
    return set(rows)


def employee_records(employee_id, start=None, end=None, limit=None):
    stmt = db.select(Attendance).where(Attendance.employee_id == int(employee_id))
    if start and end:
        stmt = stmt.where(Attendance.date >= start, Attendance.date <= end)
    stmt = stmt.order_by(Attendance.date.desc(), Attendance.timestamp.desc())
    if limit:
        stmt = stmt.limit(limit)
    return list(db.session.execute(stmt).scalars())


def salary_for(employee, present_days, period_days):
    """Monthly salary spread over calendar days of the selected period."""
    monthly = employee.monthly_salary if employee else None
    if not monthly or period_days <= 0:
        return None
    monthly = float(monthly)
    per_day = monthly / period_days
    return {
        'monthly': monthly,
        'per_day': round(per_day, 2),
        'payable': round(per_day * present_days, 2),
        'present_days': present_days,
        'period_days': period_days,
    }


# ---------------------------------------------------------------------------
# Geofencing helpers
# ---------------------------------------------------------------------------
def _haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lng points."""
    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    except (TypeError, ValueError):
        return None
    import math
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def active_locations():
    return list(
        db.session.execute(
            db.select(OfficeLocation).where(OfficeLocation.is_active == True).order_by(OfficeLocation.name)
        ).scalars()
    )


def location_within_radius(location, lat, lng):
    """Return (distance_meters, is_inside) or (None, False) when coords invalid."""
    if location is None or lat is None or lng is None:
        return None, False
    d = _haversine(location.latitude, location.longitude, lat, lng)
    if d is None:
        return None, False
    return d, d <= float(location.radius_meters)


def nearest_location(lat, lng):
    """Return (location, distance, is_inside) for the closest active location."""
    best = None
    for loc in active_locations():
        d, inside = location_within_radius(loc, lat, lng)
        if d is None:
            continue
        if best is None or d < best[1]:
            best = (loc, d, inside)
    return best if best else (None, None, False)


def on_leave_today(employee_id, date=None):
    """True if employee has an approved leave covering the given date."""
    date = date or today_date()
    req = db.session.execute(
        db.select(LeaveRequest).where(
            LeaveRequest.employee_id == employee_id,
            LeaveRequest.status == 'approved',
            LeaveRequest.start_date <= date,
            LeaveRequest.end_date >= date,
        )
    ).scalar_one_or_none()
    return req


def notice_active_today(employee_id, date=None):
    """Return active approved notice request covering the given date, or None."""
    date = date or today_date()
    return db.session.execute(
        db.select(NoticeRequest).where(
            NoticeRequest.employee_id == employee_id,
            NoticeRequest.status == 'approved',
            NoticeRequest.notice_start <= date,
            NoticeRequest.notice_end >= date,
        )
    ).scalar_one_or_none()


def can_mark_today(employee_id, date=None):
    """Return (allowed, reason). Leaves override notice for attendance blocking."""
    date = date or today_date()
    leave = on_leave_today(employee_id, date)
    if leave:
        return False, 'Aap approved leave par hain — attendance mark nahi karni.'
    notice = notice_active_today(employee_id, date)
    if notice:
        return False, f"Aap notice period ({notice.notice_start} se {notice.notice_end}) me hain — attendance mark nahi karni."
    return True, ''


def mark_present(employee, method='manual'):
    """Create today's attendance record once. Returns (created, record)."""
    today = today_date()
    existing = db.session.execute(
        db.select(Attendance).where(
            Attendance.employee_id == employee.id,
            Attendance.date == today,
        )
    ).scalar_one_or_none()
    if existing:
        return False, existing

    stamp = now_ist()
    record = Attendance(
        employee_id=employee.id,
        date=today,
        time=stamp.strftime('%H:%M:%S'),
        timestamp=stamp,
        status='Present',
        method=method,
    )
    db.session.add(record)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        existing = db.session.execute(
            db.select(Attendance).where(
                Attendance.employee_id == employee.id,
                Attendance.date == today,
            )
        ).scalar_one_or_none()
        return False, existing
    return True, record


def mark_out(employee, method='manual'):
    """Mark today's check-out once. Returns (success, record, message)."""
    today = today_date()
    record = db.session.execute(
        db.select(Attendance).where(
            Attendance.employee_id == employee.id,
            Attendance.date == today,
        )
    ).scalar_one_or_none()
    if not record:
        return False, None, 'Aaj ki check-in attendance pehle mark karein.'
    if getattr(record, 'out_time', ''):
        return False, record, f"Check-out pehle hi mark hai ({record.out_time})."
    stamp = now_ist()
    record.out_time = stamp.strftime('%H:%M:%S')
    record.out_timestamp = stamp
    record.out_method = method
    db.session.commit()
    return True, record, f"Check-out mark ho gaya — {record.out_time}."


# ---------------------------------------------------------------------------
# Public / auth routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        identifier = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        remember = bool(request.form.get('remember'))
        user = db.session.execute(
            db.select(User).where(or_(User.username == identifier, User.email == identifier))
        ).scalar_one_or_none()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f"Welcome back, {user.username}!", 'success')
            return redirect(url_for('dashboard'))
        flash('Username/email ya password galat hai.', 'error')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Aap logout ho gaye.', 'success')
    return redirect(url_for('login'))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.is_admin:
        return _admin_dashboard()
    return _employee_dashboard()


def _admin_dashboard():
    today = today_date()
    employees = list(db.session.execute(db.select(Employee).order_by(Employee.name)).scalars())
    present_ids = present_employee_ids(today)
    for employee in employees:
        employee.present = employee.id in present_ids

    total = len(employees)
    present = sum(1 for e in employees if e.present)
    stats = {
        'total': total,
        'present': present,
        'absent': max(total - present, 0),
        'rate': round(present / total * 100, 1) if total else 0,
    }

    month_start = today.replace(day=1)
    month_present = db.session.execute(
        db.select(db.func.count())
        .select_from(Attendance)
        .where(
            Attendance.date >= month_start,
            Attendance.date <= today,
            Attendance.status == 'Present',
        )
    ).scalar_one()

    return render_template(
        'dashboard.html',
        employees=employees,
        stats=stats,
        today=today.strftime('%Y-%m-%d'),
        month_present=month_present,
    )


def _employee_dashboard():
    employee = get_employee(current_user.employee_id)
    if not employee:
        flash('Aapka employee record nahi mila. Admin se contact karein.', 'error')
        return redirect(url_for('logout'))

    today = today_date()
    month_start = today.replace(day=1)
    records = employee_records(employee.id, month_start, today)
    present_days = sum(1 for r in records if r.status == 'Present')
    days_in_month = (
        today.replace(day=28) + timedelta(days=4)
    ).replace(day=1) - timedelta(days=1)
    period_days = days_in_month.day
    salary = salary_for(employee, present_days, period_days)
    today_record = db.session.execute(
        db.select(Attendance).where(
            Attendance.employee_id == employee.id,
            Attendance.date == today,
        )
    ).scalar_one_or_none()

    return render_template(
        'dashboard.html',
        employee=employee,
        today=today.strftime('%Y-%m-%d'),
        today_record=today_record,
        records=records,
        present_days=present_days,
        period_days=period_days,
        salary=salary,
    )


# ---------------------------------------------------------------------------
# Employee management (admin only)
# ---------------------------------------------------------------------------
@app.route('/add_employee', methods=['GET', 'POST'])
@admin_required
def add_employee():
    if request.method == 'POST':
        name = (request.form.get('name') or '').strip()
        employee_code = (request.form.get('employee_code') or '').strip()
        phone = (request.form.get('phone') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        salary_raw = (request.form.get('monthly_salary') or '').strip()
        username = (request.form.get('username') or '').strip() or employee_code.lower().replace(' ', '_')
        password = request.form.get('password') or 'employee123'
        photo = request.files.get('photo')

        error = None
        if not name or not employee_code:
            error = 'Name aur Employee ID zaroori hain.'
        elif not phone:
            error = 'Phone number zaroori hai (report ke liye).'
        elif db.session.execute(
            db.select(Employee).where(Employee.employee_code == employee_code)
        ).scalar_one_or_none():
            error = 'Ye Employee ID pehle se maujood hai.'
        elif db.session.execute(
            db.select(User).where(User.username == username)
        ).scalar_one_or_none():
            error = 'Ye login username pehle se maujood hai.'
        elif photo is None or not photo.filename:
            error = 'Employee ki photo upload karein.'
        elif not allowed_file(photo.filename):
            error = 'Sirf JPG/PNG photo allowed hai.'

        monthly_salary, salary_error = parse_salary(salary_raw)
        if salary_error:
            error = salary_error

        if error:
            flash(error, 'error')
            return render_template('add_employee.html', form=request.form), 400

        raw = photo.read()
        try:
            encoding = read_face_encoding(raw)
        except Exception:
            encoding = None
        if encoding is None:
            flash('Photo me chehra nahi mila. Clear front-facing photo upload karein.', 'error')
            return render_template('add_employee.html', form=request.form), 400

        photo_key, storage = _store_photo(raw, photo.mimetype)
        if photo_key is None:
            return render_template('add_employee.html', form=request.form), 502

        employee = Employee(
            name=name,
            employee_code=employee_code,
            phone=phone,
            email=email,
            monthly_salary=monthly_salary,
            photo_key=photo_key,
            photo_storage=storage,
            face_encoding=encoding.tolist(),
            active=True,
        )
        user = User(
            username=username,
            email=email or f'{username}@example.com',
            role='employee',
            employee=employee,
        )
        user.set_password(password)
        db.session.add(employee)
        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            delete_photo_key(photo_key, storage)
            flash('Employee ya username save nahi hua (duplicate ho sakta hai).', 'error')
            return render_template('add_employee.html', form=request.form), 400

        flash(f'{name} add ho gaya. Login: {username} / {password}', 'success')
        return redirect(url_for('dashboard'))

    return render_template('add_employee.html', form={})


@app.route('/employee/<int:employee_id>')
@login_required
def employee_details(employee_id):
    if not current_user.is_admin and current_user.employee_id != employee_id:
        flash('Aap sirf apni details dekh sakte hain.', 'error')
        return redirect(url_for('dashboard'))

    employee = get_employee(employee_id)
    if not employee:
        flash('Employee nahi mila.', 'error')
        return redirect(url_for('dashboard'))

    today = today_date()
    month_start = today.replace(day=1)
    last_day = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    records = employee_records(employee.id, month_start, today)
    present_days = sum(1 for r in records if r.status == 'Present')
    salary = salary_for(employee, present_days, last_day.day)
    today_record = db.session.execute(
        db.select(Attendance).where(
            Attendance.employee_id == employee.id,
            Attendance.date == today,
        )
    ).scalar_one_or_none()
    all_records = employee_records(employee.id, limit=60)

    return render_template(
        'employee_details.html',
        employee=employee,
        records=all_records,
        present_days=present_days,
        salary=salary,
        today_record=today_record,
    )


@app.route('/employee/<int:employee_id>/edit', methods=['GET', 'POST'])
@admin_required
def edit_employee(employee_id):
    employee = get_employee(employee_id)
    if not employee:
        flash('Employee nahi mila.', 'error')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name = (request.form.get('name') or '').strip()
        employee_code = (request.form.get('employee_code') or '').strip()
        phone = (request.form.get('phone') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        salary_raw = (request.form.get('monthly_salary') or '').strip()
        new_password = request.form.get('password') or ''
        photo = request.files.get('photo')

        error = None
        if not name or not employee_code:
            error = 'Name aur Employee ID zaroori hain.'
        elif not phone:
            error = 'Phone number zaroori hai.'
        elif db.session.execute(
            db.select(Employee).where(
                Employee.employee_code == employee_code,
                Employee.id != employee.id,
            )
        ).scalar_one_or_none():
            error = 'Ye Employee ID kisi aur ke paas hai.'

        monthly_salary, salary_error = parse_salary(salary_raw)
        if salary_error:
            error = salary_error

        encoding = None
        raw = b''
        if photo and photo.filename:
            if not allowed_file(photo.filename):
                error = 'Sirf JPG/PNG photo allowed hai.'
            else:
                raw = photo.read()
                try:
                    encoding = read_face_encoding(raw)
                except Exception:
                    encoding = None
                if encoding is None:
                    error = 'Nayi photo me chehra nahi mila.'

        if error:
            flash(error, 'error')
            return render_template(
                'edit_employee.html', employee=employee, form=request.form
            ), 400

        old_photo_key = employee.photo_key
        old_photo_storage = employee.photo_storage
        new_photo = None

        employee.name = name
        employee.employee_code = employee_code
        employee.phone = phone
        employee.email = email
        employee.monthly_salary = monthly_salary

        if encoding is not None:
            photo_key, storage = _store_photo(raw, photo.mimetype)
            if photo_key is None:
                db.session.rollback()
                return render_template(
                    'edit_employee.html', employee=employee, form=request.form
                ), 502
            new_photo = (photo_key, storage)
            employee.photo_key = photo_key
            employee.photo_storage = storage
            employee.face_encoding = encoding.tolist()

        if employee.user:
            employee.user.email = email or f"{employee_code.lower()}@example.com"
            if new_password:
                employee.user.set_password(new_password)

        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            if new_photo:
                employee.photo_key, employee.photo_storage = old_photo_key, old_photo_storage
                delete_photo_key(*new_photo)
            flash('Changes save nahi hue (duplicate Employee ID ho sakti hai).', 'error')
            return render_template('edit_employee.html', employee=employee, form=request.form), 400

        if new_photo:
            delete_photo_key(old_photo_key, old_photo_storage)

        flash('Employee details update ho gayi.', 'success')
        return redirect(url_for('employee_details', employee_id=employee.id))

    return render_template('edit_employee.html', employee=employee, form={})


@app.route('/employee/<int:employee_id>/delete', methods=['POST'])
@admin_required
def delete_employee(employee_id):
    employee = get_employee(employee_id)
    if not employee:
        flash('Employee nahi mila.', 'error')
        return redirect(url_for('dashboard'))

    name = employee.name
    delete_photo(employee)
    db.session.delete(employee)  # cascades to user + attendance
    db.session.commit()
    flash(f"{name} aur unka data delete ho gaya.", 'success')
    return redirect(url_for('dashboard'))


# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------
@app.route('/mark_attendance', methods=['GET'])
@employee_required
def mark_attendance_page():
    employee = get_employee(current_user.employee_id)
    if not employee:
        flash('Aapka employee record nahi mila.', 'error')
        return redirect(url_for('logout'))
    today = today_date()
    today_record = db.session.execute(
        db.select(Attendance).where(
            Attendance.employee_id == employee.id,
            Attendance.date == today,
        )
    ).scalar_one_or_none()
    records = employee_records(employee.id, limit=15)
    return render_template(
        'mark_attendance.html',
        employee=employee,
        today_record=today_record,
        locations=active_locations(),
        records=records,
    )


@app.route('/mark_attendance', methods=['POST'])
@employee_required
def mark_attendance():
    employee = get_employee(current_user.employee_id)
    if not employee:
        return jsonify({'success': False, 'message': 'Employee record nahi mila.'}), 404

    # Leave ya notice period me attendance block karein
    allowed, reason = can_mark_today(employee.id)
    if not allowed:
        return jsonify({'success': False, 'message': reason}), 403

    payload = request.get_json(silent=True) or {}
    action = (payload.get('action') or 'in').strip().lower()
    if action not in ('in', 'out'):
        return jsonify({'success': False, 'message': 'Invalid attendance action.'}), 400
    method = 'manual'

    # ---------- Geofencing (sirf check-in par enforce hota hai) ----------
    lat = lng = None
    location_id = None
    locations = active_locations()
    if locations:
        try:
            lat = float(payload.get('lat')) if payload.get('lat') not in (None, '') else None
            lng = float(payload.get('lng')) if payload.get('lng') not in (None, '') else None
        except (TypeError, ValueError):
            lat = lng = None
        if lat is None or lng is None:
            return jsonify({
                'success': False,
                'message': 'Attendance ke liye location access zaroori hai. Browser se location permission dein aur dobara koshish karein.',
            }), 400
        nearest, distance, inside = nearest_location(lat, lng)
        if not inside:
            return jsonify({
                'success': False,
                'message': f'Aap office location ke radius ke bahar hain ({distance:.0f} meter door). Sirf office ke andar se attendance mark kar sakte hain.',
                'outside_radius': True,
                'distance': round(distance, 1),
            }), 400
        location_id = nearest.id if nearest else None

    if not payload.get('manual'):
        image_data = payload.get('image')
        if not image_data:
            return jsonify({'success': False, 'message': 'Koi image nahi mili.'}), 400
        method = 'face'
        try:
            encoded = image_data.split(',', 1)[1]
            raw = base64.b64decode(encoded)
            image_np = np.array(Image.open(io.BytesIO(raw)).convert('RGB'))
            fr = _get_face_recognition()
            encodings = fr.face_encodings(image_np)
        except Exception:
            return jsonify({'success': False, 'message': 'Image process nahi ho payi.'}), 400

        if not encodings:
            return jsonify({'success': False, 'message': 'Camera me chehra nahi mila.'})

        if not employee.face_encoding:
            return jsonify(
                {'success': False, 'message': 'Aapki face encoding save nahi hai. Admin se contact karein.'}
            )

        known = np.array(employee.face_encoding)
        distances = fr.face_distance([known], encodings[0])
        if len(distances) == 0 or distances[0] > 0.5:
            return jsonify(
                {
                    'success': False,
                    'message': 'Chehra aapke record se match nahi hua. Sirf apni attendance mark kar sakte hain.',
                }
            )

    if action == 'out':
        ok, record, message = mark_out(employee, method)
        return jsonify({
            'success': ok,
            'message': message,
            'time': (record.out_time if record else '') or '',
            'method': method,
            'action': 'out',
        }), (200 if ok else 400)

    created, record = mark_present(employee, method)
    if not created:
        return jsonify(
            {
                'success': False,
                'message': f"Aaj ki check-in pehle hi mark hai ({record.time if record else ''}).",
                'action': 'in',
            }
        )

    # Check-in location save karein
    if lat is not None:
        record.check_in_lat = str(lat)
        record.check_in_lng = str(lng)
    record.location_id = location_id
    db.session.commit()

    return jsonify(
        {
            'success': True,
            'message': f"Check-in mark ho gayi — {record.time} ({'face' if method == 'face' else 'manual'}).",
            'time': record.time,
            'method': method,
            'action': 'in',
        }
    )


@app.route('/attendance_report')
@login_required
def attendance_report():
    today = today_date()
    default_start = today.replace(day=1)
    default_end = today
    start_date, end_date = default_start, default_end

    if request.args.get('start_date') or request.args.get('end_date'):
        start_date = parse_date(request.args.get('start_date'), default_start)
        end_date = parse_date(request.args.get('end_date'), default_end)

    if start_date > end_date:
        start_date, end_date = end_date, start_date
    start_s, end_s = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    period_days = (end_date - start_date).days + 1

    is_admin = current_user.is_admin
    filter_employee = (request.args.get('employee_id') or '').strip()

    stmt = db.select(Attendance).where(
        Attendance.date >= start_date,
        Attendance.date <= end_date,
    )
    if is_admin:
        if filter_employee:
            try:
                stmt = stmt.where(Attendance.employee_id == int(filter_employee))
            except ValueError:
                filter_employee = ''
        employees = list(
            db.session.execute(db.select(Employee).order_by(Employee.name)).scalars()
        )
    else:
        stmt = stmt.where(Attendance.employee_id == int(current_user.employee_id))
        own = get_employee(current_user.employee_id)
        employees = [own] if own else []

    stmt = stmt.order_by(Attendance.date.desc(), Attendance.timestamp.desc())
    records = list(db.session.execute(stmt).scalars())

    present_by_employee = {}
    for record in records:
        if record.status == 'Present':
            present_by_employee[record.employee_id] = present_by_employee.get(record.employee_id, 0) + 1

    salary_rows = []
    for employee in employees:
        present_days = present_by_employee.get(employee.id, 0)
        salary = salary_for(employee, present_days, period_days)
        salary_rows.append(
            {
                'employee': employee,
                'present_days': present_days,
                'salary': salary,
            }
        )

    total_present = sum(row['present_days'] for row in salary_rows)
    payable_total = sum(
        row['salary']['payable'] for row in salary_rows if row['salary']
    )
    stats = {
        'employees': len(employees),
        'present_records': total_present,
        'period_days': period_days,
        'payable_total': payable_total,
    }

    return render_template(
        'attendance_report.html',
        records=records,
        salary_rows=salary_rows,
        stats=stats,
        employees=employees if is_admin else [],
        is_admin=is_admin,
        start_date=start_s,
        end_date=end_s,
        filter_employee=filter_employee,
    )


@app.route('/download_attendance')
@login_required
def download_attendance():
    start_s = request.args.get('start_date')
    end_s = request.args.get('end_date')

    stmt = db.select(Attendance)
    if start_s and end_s:
        start_date = parse_date(start_s, None)
        end_date = parse_date(end_s, None)
        if start_date and end_date:
            stmt = stmt.where(Attendance.date >= start_date, Attendance.date <= end_date)
    if current_user.is_admin:
        employee_id = request.args.get('employee_id')
        if employee_id:
            try:
                stmt = stmt.where(Attendance.employee_id == int(employee_id))
            except ValueError:
                pass
    else:
        stmt = stmt.where(Attendance.employee_id == int(current_user.employee_id))

    stmt = stmt.order_by(Attendance.date.asc(), Attendance.timestamp.asc())
    records = list(db.session.execute(stmt).scalars())

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Employee ID', 'Name', 'Phone', 'Check-In Time', 'Check-Out Time', 'Status', 'Check-In Method', 'Check-Out Method'])
    for record in records:
        employee = record.employee
        writer.writerow(
            [
                record.date.strftime('%Y-%m-%d') if record.date else '',
                employee.employee_code if employee else '',
                employee.name if employee else '',
                employee.phone if employee else '',
                record.time or '',
                getattr(record, 'out_time', '') or '',
                record.status or '',
                record.method or '',
                getattr(record, 'out_method', '') or '',
            ]
        )

    filename = f"attendance_{start_s or 'all'}_{end_s or 'all'}.csv"
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'},
    )


@app.route('/delete_attendance/<int:record_id>', methods=['DELETE'])
@admin_required
def delete_attendance(record_id):
    record = db.session.get(Attendance, record_id)
    if not record:
        return jsonify({'success': False, 'message': 'Record nahi mila.'}), 404
    db.session.delete(record)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Attendance record delete ho gaya.'})


# --------------------------------------------------------------------------- #
# Office locations (admin — geofencing, max 5)
# --------------------------------------------------------------------------- #
MAX_LOCATIONS = 5


@app.route('/admin/locations', methods=['GET'])
@admin_required
def admin_locations():
    locations = list(
        db.session.execute(
            db.select(OfficeLocation).order_by(OfficeLocation.name)
        ).scalars()
    )
    return render_template(
        'office_locations.html',
        locations=locations,
        locations_json=[loc.to_dict() for loc in locations],
        MAX_LOCATIONS=MAX_LOCATIONS,
    )


@app.route('/admin/locations', methods=['POST'])
@admin_required
def add_location():
    name = (request.form.get('name') or '').strip()
    lat_raw = (request.form.get('latitude') or '').strip()
    lng_raw = (request.form.get('longitude') or '').strip()
    radius_raw = (request.form.get('radius_meters') or '100').strip()

    if not name or not lat_raw or not lng_raw:
        flash('Name, latitude aur longitude sab zaroori hain.', 'error')
        return redirect(url_for('admin_locations'))
    try:
        latitude = float(lat_raw)
        longitude = float(lng_raw)
        radius = int(radius_raw)
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError
        if radius <= 0:
            raise ValueError
    except ValueError:
        flash('Valid latitude, longitude aur radius (meter) dalein.', 'error')
        return redirect(url_for('admin_locations'))

    count = db.session.execute(db.select(db.func.count(OfficeLocation.id))).scalar() or 0
    if count >= MAX_LOCATIONS:
        flash(f'Adhiktam {MAX_LOCATIONS} office locations add ki ja sakti hain.', 'error')
        return redirect(url_for('admin_locations'))

    loc = OfficeLocation(
        name=name,
        latitude=latitude,
        longitude=longitude,
        radius_meters=radius,
        is_active=True,
    )
    db.session.add(loc)
    db.session.commit()
    flash(f'Location "{name}" add ho gayi ({radius}m radius).', 'success')
    return redirect(url_for('admin_locations'))


@app.route('/admin/locations/<int:loc_id>/toggle', methods=['POST'])
@admin_required
def toggle_location(loc_id):
    loc = db.session.get(OfficeLocation, loc_id)
    if not loc:
        flash('Location nahi mila.', 'error')
        return redirect(url_for('admin_locations'))
    loc.is_active = not loc.is_active
    db.session.commit()
    flash(f'Location {"active" if loc.is_active else "inactive"} ho gayi.', 'success')
    return redirect(url_for('admin_locations'))


@app.route('/admin/locations/<int:loc_id>/delete', methods=['POST'])
@admin_required
def delete_location(loc_id):
    loc = db.session.get(OfficeLocation, loc_id)
    if not loc:
        flash('Location nahi mila.', 'error')
        return redirect(url_for('admin_locations'))
    db.session.delete(loc)
    db.session.commit()
    flash('Location delete ho gayi.', 'success')
    return redirect(url_for('admin_locations'))


# --------------------------------------------------------------------------- #
# Leave requests
# --------------------------------------------------------------------------- #
@app.route('/leave', methods=['GET', 'POST'])
@login_required
def leave_page():
    employee = get_employee(current_user.employee_id)
    if not employee:
        flash('Employee record nahi mila.', 'error')
        return redirect(url_for('logout'))

    if request.method == 'POST':
        start_date = parse_date(request.form.get('start_date'), None)
        end_date = parse_date(request.form.get('end_date'), None)
        reason = (request.form.get('reason') or '').strip()
        if not start_date or not end_date:
            flash('Start aur end date dono zaroori hain.', 'error')
        elif end_date < start_date:
            flash('End date start date se pehle nahi ho sakti.', 'error')
        else:
            req = LeaveRequest(
                employee_id=employee.id,
                start_date=start_date,
                end_date=end_date,
                reason=reason,
            )
            db.session.add(req)
            db.session.commit()
            flash('Leave request bhej di gayi. Admin approval ka intezar karein.', 'success')
            return redirect(url_for('leave_page'))

    my_leaves = list(
        db.session.execute(
            db.select(LeaveRequest)
            .where(LeaveRequest.employee_id == employee.id)
            .order_by(LeaveRequest.created_at.desc())
        ).scalars()
    )
    return render_template('leave.html', employee=employee, my_leaves=my_leaves)


@app.route('/admin/leave', methods=['GET'])
@admin_required
def admin_leave():
    leaves = list(
        db.session.execute(db.select(LeaveRequest).order_by(LeaveRequest.created_at.desc())).scalars()
    )
    return render_template('admin_leave.html', leaves=leaves)


@app.route('/admin/leave/<int:leave_id>/<action>', methods=['POST'])
@admin_required
def admin_leave_action(leave_id, action):
    if action not in ('approve', 'reject'):
        flash('Invalid action.', 'error')
        return redirect(url_for('admin_leave'))
    req = db.session.get(LeaveRequest, leave_id)
    if not req:
        flash('Leave request nahi mila.', 'error')
        return redirect(url_for('admin_leave'))
    req.status = 'approved' if action == 'approve' else 'rejected'
    req.admin_id = current_user.id
    req.reviewed_at = now_ist()
    req.admin_note = (request.form.get('note') or '').strip()
    db.session.commit()
    flash(f'Leave request {req.status}.', 'success')
    return redirect(url_for('admin_leave'))


# --------------------------------------------------------------------------- #
# Notice period requests
# --------------------------------------------------------------------------- #
@app.route('/notice', methods=['GET', 'POST'])
@login_required
def notice_page():
    employee = get_employee(current_user.employee_id)
    if not employee:
        flash('Employee record nahi mila.', 'error')
        return redirect(url_for('logout'))

    if request.method == 'POST':
        start_date = parse_date(request.form.get('notice_start'), None)
        months_raw = (request.form.get('notice_months') or '').strip()
        reason = (request.form.get('reason') or '').strip()
        if not start_date or not months_raw:
            flash('Notice start date aur notice period (months) zaroori hain.', 'error')
            return redirect(url_for('notice_page'))
        try:
            months = int(months_raw)
            if months <= 0:
                raise ValueError
        except ValueError:
            flash('Notice period ek valid number (months me) hona chahiye.', 'error')
            return redirect(url_for('notice_page'))
        notice_end = start_date + timedelta(days=months * 30)
        req = NoticeRequest(
            employee_id=employee.id,
            notice_start=start_date,
            notice_end=notice_end,
            reason=reason,
        )
        db.session.add(req)
        db.session.commit()
        flash(f'Notice period request bhej di gayi ({months} month). Admin approval ka intezar karein.', 'success')
        return redirect(url_for('notice_page'))

    my_notices = list(
        db.session.execute(
            db.select(NoticeRequest)
            .where(NoticeRequest.employee_id == employee.id)
            .order_by(NoticeRequest.created_at.desc())
        ).scalars()
    )
    return render_template('notice.html', employee=employee, my_notices=my_notices)


@app.route('/admin/notice', methods=['GET'])
@admin_required
def admin_notice():
    notices = list(
        db.session.execute(db.select(NoticeRequest).order_by(NoticeRequest.created_at.desc())).scalars()
    )
    return render_template('admin_notice.html', notices=notices)


@app.route('/admin/notice/<int:notice_id>/<action>', methods=['POST'])
@admin_required
def admin_notice_action(notice_id, action):
    if action not in ('approve', 'reject'):
        flash('Invalid action.', 'error')
        return redirect(url_for('admin_notice'))
    req = db.session.get(NoticeRequest, notice_id)
    if not req:
        flash('Notice request nahi mila.', 'error')
        return redirect(url_for('admin_notice'))
    req.status = 'approved' if action == 'approve' else 'rejected'
    req.admin_id = current_user.id
    req.reviewed_at = now_ist()
    req.admin_note = (request.form.get('note') or '').strip()
    db.session.commit()
    flash(f'Notice period request {req.status}.', 'success')
    return redirect(url_for('admin_notice'))


# ---------------------------------------------------------------------------
# Local photo serving (used when R2 is not configured)
# ---------------------------------------------------------------------------
@app.route('/uploads/<path:filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
