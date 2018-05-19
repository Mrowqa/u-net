import datetime


def file_formatted_now():
    return datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
