from django.contrib import admin

# Register your models here.
from .models import *

admin.site.register(user)
admin.site.register(file)
admin.site.register(MLmodel)
