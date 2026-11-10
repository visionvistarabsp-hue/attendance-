import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db, OfficeLocation, User, LeaveRequest, NoticeRequest, Employee, Attendance, mark_present, mark_out, can_mark_today, on_leave_today, nearest_location, location_within_radius
from werkzeug.security import generate_password_hash
from datetime import date, timedelta


with app.app_context():
    db.drop_all()
    db.create_all()
    admin = User(username='admin', email='admin@example.com', role='admin')
    admin.password_hash = generate_password_hash('admin123')
    db.session.add(admin)
    emp = Employee(name='Test Employee', employee_code='EMP-001', phone='9999999999', email='e@e.com')
    db.session.add(emp)
    db.session.commit()
    eid = emp.id
    loc = OfficeLocation(name="Test's Office", latitude=28.6139, longitude=77.2090, radius_meters=100, is_active=True)
    db.session.add(loc)
    db.session.commit()
    u = User(username='emp1', email='emp1@e.com', role='employee')
    u.password_hash = generate_password_hash('emp123')
    u.employee_id = eid
    db.session.add(u)
    db.session.commit()

with app.test_client() as c:
    c.post('/login', data={'username': 'admin', 'password': 'admin123'})
    with app.app_context():
        db.session.query(LeaveRequest).delete()
        db.session.commit()

    c.get('/logout')
    c.post('/login', data={'username': 'emp1', 'password': 'emp123'})

    # inside radius
    resp = c.post('/mark_attendance', json={'manual': True, 'action': 'in', 'lat': 28.6140, 'lng': 77.2091})
    print('Check-in inside:', resp.status_code, resp.get_json())

    resp = c.post('/mark_attendance', json={'manual': True, 'action': 'out', 'lat': 28.6140, 'lng': 77.2091})
    print('Check-out inside:', resp.status_code, resp.get_json())

    # outside radius
    resp = c.post('/mark_attendance', json={'manual': True, 'action': 'in', 'lat': 28.6200, 'lng': 77.2200})
    print('Check-in outside:', resp.status_code, resp.get_json())

    # leave blocking
    with app.app_context():
        today = date.today()
        leave = LeaveRequest(employee_id=eid, start_date=today, end_date=today, reason='sick', status='approved')
        db.session.add(leave)
        db.session.commit()
    resp = c.post('/mark_attendance', json={'manual': True, 'action': 'in', 'lat': 28.6140, 'lng': 77.2091})
    print('Check-in on leave:', resp.status_code, resp.get_json())

    with app.app_context():
        att = db.session.get(Attendance, 1)
        print('Stored loc_id:', att.location_id, 'lat:', att.check_in_lat, 'lng:', att.check_in_lng)

print('DONE')