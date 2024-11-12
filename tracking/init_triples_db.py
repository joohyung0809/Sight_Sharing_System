import sqlite3

# 데이터베이스 연결 (없으면 triple_s.db 파일을 생성합니다)
conn = sqlite3.connect('triple_s.db')
cursor = conn.cursor()

# detections 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    object_type VARCHAR(20) NOT NULL,
    x_coordinate FLOAT NOT NULL,
    y_coordinate FLOAT NOT NULL,
    direction VARCHAR(20)
);
''')

# signals 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS signals (
    signal_id INTEGER PRIMARY KEY,
    signal_type VARCHAR(20) NOT NULL,
    timestamp DATETIME NOT NULL,
    current_state VARCHAR(10) NOT NULL,
    led_color VARCHAR(10) NOT NULL,
    sound_active BOOLEAN NOT NULL
);
''')

# detector_status 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS detector_status (
    detector_id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    ap_mode_active BOOLEAN NOT NULL,
    connected_signals INTEGER NOT NULL,
    current_risk_level VARCHAR(10) NOT NULL
);
''')

# risk_events 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS risk_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    detector_id INTEGER NOT NULL,
    risk_level VARCHAR(10) NOT NULL,
    vehicle_count INTEGER,
    pedestrian_count INTEGER,
    description TEXT,
    FOREIGN KEY (detector_id) REFERENCES detector_status(detector_id)
);
''')

# 변경사항 커밋 및 연결 종료
conn.commit()
conn.close()

print("데이터베이스와 테이블이 성공적으로 생성되었습니다.")