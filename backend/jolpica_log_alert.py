import time
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

LOG_FILE = 'backend.log'
PATTERN = re.compile(r'\[Jolpica\] New or unexpected')
EMAIL_TO = 'varunsuk@umich.edu'
EMAIL_FROM = 'f1.alerts@yourdomain.com'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USER = 'f1.alerts@yourdomain.com'
SMTP_PASS = 'your_app_password_here'

SUBJECT = 'Jolpica Alert: New or Unexpected Field Detected'


def send_email_alert(message):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO
    msg['Subject'] = SUBJECT
    msg.attach(MIMEText(message, 'plain'))
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        server.quit()
        print(f"Alert sent: {message}")
    except Exception as e:
        print(f"Failed to send alert: {e}")

def monitor_log():
    with open(LOG_FILE, 'r') as f:
        f.seek(0, 2)  # Go to end of file
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            if PATTERN.search(line):
                send_email_alert(f'Jolpica Alert: {line.strip()}')

if __name__ == '__main__':
    monitor_log() 