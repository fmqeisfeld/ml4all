from django.db import models
from django.conf import settings


import os 
# linter deaktiviert: str+shift+p -> anaconda -> disabel ... this file

#file validation stuff
#https://www.jeremyaldrich.net/en/latest/django_filefield_csv_validation.html
import csv
from django.core.exceptions import ValidationError
from django.utils.translation import ugettext_lazy as _
import magic

# for custom registration and e-mail validation
# quelle: https://simpleisbetterthancomplex.com/tutorial/2017/02/18/how-to-create-user-sign-up-view.html
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

# make Email unique !
User._meta.get_field('email')._unique = True

# extend User model with custom user model
class user(models.Model):
    #email = models.EmailField(max_length=200, null=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE, null =True)
    email_confirmed = models.BooleanField(default=False)    

    date_created = models.DateTimeField(auto_now_add=True, null=True)

    running_jobs = models.IntegerField(default=0)
    ################
    # CONFIGS
    ################

    #### LOGISTIC REGRESSION
    choices_logreg_maxiter = ( 
                (1000,'1000'),
                (10000,'10000'),
                (100000,'100000'),
                (-1,'Bis Konvergenz (*)'),
    )
    choices_logreg_testsize = (
        (0.1,'10%'),
        (0.33,'33%'),
        (0.5,'50%'),
    )
    choices_logreg_optimloops = (
        (0,'Keine'),
        (-1,'Bis Konvergenz (*)')
    )


    logreg_maxiter =  models.IntegerField(default=100000,choices=choices_logreg_maxiter)
    logreg_optimloops = models.IntegerField(default=0,choices=choices_logreg_optimloops)
    logreg_testsize = models.FloatField(default=0.33,choices=choices_logreg_testsize)
    logreg_C = models.FloatField(default=1.0,)

    #### SVC #####
    choices_svc_kernel = (('linear','Linear'),
                           ('poly','Polynom'),
                           ('rbf','Radiale Basisfunktion (*)'),
                         )
    choices_svc_degree = ((2,2), (3,3),(4,4))

    svc_maxiter =  models.IntegerField(default=100000,choices=choices_logreg_maxiter)
    svc_optimloops = models.IntegerField(default=0,choices=choices_logreg_optimloops)
    svc_testsize = models.FloatField(default=0.33,choices=choices_logreg_testsize)
    svc_C = models.FloatField(default=1.0)
    svc_kernel = models.CharField(default='linear',choices=choices_svc_kernel, max_length=20)
    svc_degree = models.IntegerField(default=3, choices=choices_svc_degree)

    ##### SVR ######
    svr_maxiter =  models.IntegerField(default=100000,choices=choices_logreg_maxiter)
    svr_optimloops = models.IntegerField(default=0,choices=choices_logreg_optimloops)
    svr_testsize = models.FloatField(default=0.33,choices=choices_logreg_testsize)
    svr_C = models.FloatField(default=1.0)
    svr_kernel = models.CharField(default='poly',choices=choices_svc_kernel, max_length=20)
    svr_degree = models.IntegerField(default=3, choices=choices_svc_degree)

    #### ARIMA ####
    choices_arima_pq = ((5,5) , (40,'40 (*)'))
    choices_arima_forecast = ((0.1,'10%') , 
                              (0.2,'20%'),
                              (0.5,'50%'))

    choices_arima_optimloops  = ((2,2),(-1,'Alle (*)'))                            

    arima_maxpq = models.IntegerField(default=5,choices=choices_arima_pq)
    arima_testsize = models.FloatField(default=0.33,choices=choices_logreg_testsize)
    arima_forecast = models.FloatField(default=0.33,choices=choices_arima_forecast)
    arima_optimloops = models.IntegerField(default=2,choices=choices_arima_optimloops)

    #### LGBM ####
    choices_lgbm_optimloops  = ((5,5),(-1,'Bis Konvergenz (*)'))  

    lgbm_testsize = models.FloatField(default=0.33,choices=choices_logreg_testsize)
    lgbm_forecast = models.FloatField(default=0.33,choices=choices_arima_forecast)
    lgbm_optimloops = models.IntegerField(default=5,choices=choices_lgbm_optimloops)

    def __str__(self):
        return self.user.email

@receiver(post_save, sender=User)
def update_user_profile(sender, instance, created, **kwargs):
    if created:
        user.objects.create(user=instance)
    instance.user.save()

##########################
# Helper functions
########################
def user_dir_path(instance,filename):
    #fp = '{0}user_{1}/{2}'.format(settings.MEDIA_ROOT,str(instance.user.id),filename)
    #war vorher nicht korrekt!
    # The upload_to callable should return a path that is relative to the MEDIA_ROOT because it will be appended to it. 
    #siehe https://stackoverflow.com/questions/50819578/wrong-path-for-imagefield-in-django
    #
    #dazu in settings.py MEDIA_ROOT und MEDIA URL definieren und der zweiten urls.py registrieren!
    fp = 'user_{0}/{1}'.format(str(instance.user.id),filename)
    return fp
    #return settings.MEDIA_ROOT+'foo.csv'

def import_document_validator(document):
    max_size = 1000000 # 1mb # 128000 #byte -> 128 kb

    infilename=document.name
    infile=document.file

    extension = os.path.splitext(infilename)[1]

    valid_extensions = ['.csv','.xlsx', '.xls', '.ods', '.odf']
    if not extension.lower() in valid_extensions:
        raise ValidationError('Fehler: Es werden nur Dateien vom Typ CSV, XLS sowie XLSX, ODF und ODS akzeptiert.')    


    valid_mime_types = ['application/vnd.ms-excel', # ms excel
                        'application/zip', #ebenfalls excel
                        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', #open office
                        'text/plain', # csv
                        'application/csv',
                        'application/vnd.oasis.opendocument.spreadsheet', # ODS
                       ]
    mime_type = magic.from_buffer(infile.read(1024), mime=True)

    if mime_type not in valid_mime_types:
        raise ValidationError("""Fehler: Es werden nur Dateien vom Typ CSV, XLS sowie XLSX, ODF und ODS akzeptiert.
                                 Bei der gegebenen Datei handelt es sich jedoch um den Typ:{}""".format(mime_type))

    if mime_type =='text/plain':
        try:
            #ACHTUNG: wenn zu wenig eingelesen wird, kann evtl. sniffer probleme machen!
            sample = infile.read(2*2048).decode('utf-8') # ACHTUNG: Wird fehler produzieren falls zu viele header-strings
                                                       # z.B. bei sehr langer benennung einzelner cols
            dialect = csv.Sniffer().sniff(sample,delimiters=',')
            infile.seek(0, 0)
        except csv.Error as e:
            raise ValidationError('Fehler: Die CSV-Datei ist ungültig:' + str(e))

    #check file size
    infile.seek(0,2)
    fsize=infile.tell()
    if fsize > max_size:
        raise ValidationError('Fehler: Die maximale Größe der Dateien ist auf 1 MB beschränkt.')                              

    return True  

######################
class file(models.Model):

    validators_list = [import_document_validator]
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    path = models.FileField(upload_to=user_dir_path,validators=validators_list)
                            #widget=forms.FileInput(attrs={'accept':['.csv','.xls','xlsx']}))    

    #filename = models.CharField("filename",max_length=30)
    date_created = models.DateTimeField(auto_now_add=True, null=True)
    rows = models.IntegerField(null=True)    

    # dataset fields
    # comma-separated content (per column)
    header=  models.CharField(max_length=120,default='')
    histo = models.BinaryField(default=None, null=True)
    counts = models.CharField(max_length=120,default='')
    minvals= models.CharField(max_length=120,default='')
    maxvals= models.CharField(max_length=120,default='')
    medvals= models.CharField(max_length=120,default='')
    cols   = models.CharField(max_length=120,default='')
    types  = models.CharField(max_length=120,default='')


    
    def __str__(self):
        print(str(os.path.basename(str(self.path))))
        return str(os.path.basename(str(self.path)))


    def delete(self, *args, **kwargs):
        self.path.delete()
        super().delete(*args, **kwargs)        


#############################
class MLmodel(models.Model):
    #The first element in each tuple is the actual value to be set on the model, and the second element is the human-readable name

    choices = ( 
                ('FC','Entscheidungsbaum Klassifizierung'),
                ('FR','Entscheidungsbaum Regression'),
                ('LR','Logistische Regression'),
                ('KC','Clusteranalysie'),
                ('AN','Anomalie'),
                ('AS','Assoziation'),
                ('KO','Korrelation'),
                ('ARI','Arima Prognose'),
                ('GBM','LightGBM Prognose'),
                ('SVC','Stützvektormaschine Klassifizierung'),
                ('SVR','Stützvektormaschine Regression'),
    )
    user =    models.ForeignKey(User,on_delete=models.CASCADE)
    dataset = models.ForeignKey(file, on_delete=models.CASCADE,related_name='dataset')

    model_type = models.CharField(max_length=50,choices=choices)
    results = models.FileField(upload_to=user_dir_path)
    date_created = models.DateTimeField(auto_now_add=True, null=True)    
    model_file = models.FileField(upload_to = user_dir_path, null=True, blank=True ) # saved in pkl format
    download_file = models.FileField(upload_to = user_dir_path, null=True, blank=True ) # csv format. manually delete before new eval

    # Zielgröße
    objective = models.CharField(max_length=120,default='')

    def __str__(self):
        return (self.dataset).__str__()

    def delete(self, *args, **kwargs):
        self.results.delete()
        self.model_file.delete()
        self.download_file.delete()
        super().delete(*args, **kwargs)  



