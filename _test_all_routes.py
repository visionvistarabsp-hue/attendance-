import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db, OfficeLocation, User, LeaveRequest, NoticeRequest, Employee
from werkzeug.security import generate_password_hash
from datetime import date, timedelta


def reset():
    db.drop_all()
    db.create_all()
    admin = User(username='admin', email='admin@example.com', role='admin')
    admin.password_hash = generate_password_hash('admin123')
    db.session.add(admin)
    emp = Employee(name='Test Employee', employee_code='EMP-001', phone='9999999999', email='e@e.com')
    db.session.add(emp)
    db.session.commit()
    admin.employee_id = None
    emp.user = None
    db.session.commit()
    return admin.id, emp.id


with app.app_context():
    admin_id, emp_id = reset()
    # add a location
    loc = OfficeLocation(name="Test's Office", latitude=28.6139, longitude=77.2090, radius_meters=100, is_active=True)
    db.session.add(loc)
    db.session.commit()

with app.test_client() as c:
    c.post('/login', data={'username': 'admin', 'password': 'admin123'})

    for path in ['/admin/locations', '/admin/leave', '/admin/notice']:
        r = c.get(path)
        print(path, r.status_code)
        if r.status_code != 200:
            print(r.data.decode('utf-8', errors='replace')[:1500])

    # employee leave page
    c.get('/logout')
    c.post('/login', data={'username': 'admin', 'password': 'admin123'})
    # create employee user
    with app.app_context():
        u = User(username='emp1', email='emp1@e.com', role='employee')
        u.password_hash = generate_password_hash('emp123')
        db.session.add(u)
        db.session.commit()
        eid = db.session.get(Employee, 1)
        u.employee_id = eid.id
        db.session.commit()
        uid = u.id
    c.get('/logout')
    c.post('/login', data={'username': 'emp1', 'password': 'emp123'})
    for path in ['/leave', '/notice', '/mark_attendance']:
        r = c.get(path)
        print(path, r.status_code)
        if r.status_code != 200:
            print(r.data.decode('utf-8', errors='replace')[:1500])

print('DONE')