import pytest  
import django  
from django.conf import settings  
  
def pytest_configure():  
    settings.configure(  
        DEBUG=True,  
        DATABASES={  
            'default': {  
                'ENGINE': 'django.db.backends.sqlite3',  
                'NAME': ':memory:',  
            }  
        },  
        INSTALLED_APPS=[  
            'django.contrib.auth',  
            'django.contrib.contenttypes',  
            'authapp',  
            'csv_anonymizer',  
            'recommendation_engine',  
        ],  
    )  
    django.setup()