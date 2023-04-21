import requests
SECRET_URL = None
SECRET_USER = None

def login(url, user):
    global SECRET_URL, SECRET_USER
    SECRET_URL = url
    SECRET_USER = user

def notify(message):
    assert SECRET_URL is not None
    assert SECRET_USER is not None
    requests.post(SECRET_URL, json={
        'text': f'{message}',
        'blocks': [{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"<@{SECRET_USER}> {message}"
            }
        }]
    })