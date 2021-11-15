import sys
from datetime import datetime
from functools import wraps
from microservices.models.interface import Response

def response_decorator(func):
    # job = func.__name__
    @wraps(func)
    def wrapper(*args, **kwargs):
        response = Response()
        # response.job = job
        execution_begin = datetime.now()
        response.execution_begin = execution_begin
        response.status = True

        try:
            output = func(*args, **kwargs) # decorated function
            response.output = output
        except Exception as error:
            response.status = False
            response.error_message = "{name}: {message}".format(name=type(error), message=str(error))

        execution_end = datetime.now()
        execution_time = (execution_end - execution_begin).total_seconds() # total execution time
        response.execution_end = execution_end
        response.execution_time = execution_time
        return response
    return wrapper