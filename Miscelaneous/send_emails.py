import creds
import smtplib
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def email_new(df):
    message = MIMEMultipart()
    message['Subject'] = 'Flagged Data'
    message['From'] = creds.sender
    message['To'] = creds.recipient
    message.preamble = 'This is a test email'
    
    html = MIMEText(df.to_html(index=False), "html")
    message.attach(html)
    with smtplib.SMTP("smtp.outlook.com", 587) as server:
        server.starttls()
        server.login(creds.sender, creds.password)
        server.sendmail(creds.sender, creds.recipient, message.as_string())
        
        
        
if __name__ == '__main__':
    df = pd.DataFrame({'Test': [1,2,3], 'Data':[123, '4124','ADDas']})
    email_new(df)