from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.http import JsonResponse
from django.contrib.auth.mixins import LoginRequiredMixin                                            
from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.views.generic import TemplateView, FormView
from .models import file, user, MLmodel  
from .forms import UploadForm
from django.urls import reverse_lazy
from django.core.files import File # wrapper for djangos file format

# for filetype detection
import magic
import odf # zum lesen von odf, ods  ('OpenDocument Spreadsheet')

#########################
# for registration      #
#########################
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from .forms import SignUpForm, ConfigForm

from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode
from django.template.loader import render_to_string
from .tokens import account_activation_token
from django.utils.encoding import force_text


from django.contrib.auth.models import User
from django.utils.http import urlsafe_base64_decode

###########################
############################
from django.conf import settings
#from django.core.exceptions import ValidationError
#from django.core.files.storage import default_storage # for validation
#import csv #for validation
################################
import os 
import pandas as pd
import numpy as np
import pickle
import base64
#################################
# for time-series
from pandas.core.tools.datetimes import _guess_datetime_format_for_array
##################################
from . import myml

#import time 
# Create your views here.

class LandingView(View):
    def get(self,request,stay=None):
        #message = render_to_string('accounts/activmail.html')        

        if request.user.is_authenticated and stay != 'start':        
            #return HttpResponseRedirect('dashboard')  
            return redirect(reverse_lazy('dashboard'))
        else:
            return render(request,'accounts/landing.html')
            


class ConfigView(LoginRequiredMixin,TemplateView):    
    def post(self, request, *args, **kwargs):
        # user ist eine extension vom eigentlichen USER
        # sieht etwas verwirrend aus, weil die User-ID im request in "user" (klein geschrieben)
        # hinterlegt ist
        userinstance = user.objects.get(user=request.user.pk)
        # wichtig: userinstance mitgeben sonst wird nicht gespeichert
        # stattdessen wird neue instanz erzeugt 
        form = ConfigForm(request.POST,instance=userinstance) 

        if form.is_valid():
            form.save()
            return render(request,'accounts/config.html',{'form': form, 'success': True})
        else:
            return render(request,'accounts/config.html',{'form': form})
        
    def get(self,request):
        # 
        userinstance = user.objects.get(user=request.user.pk)
        form = ConfigForm(instance= userinstance)
        return render(request,'accounts/config.html',{'form': form})


def Registration(request):
    template = 'accounts/info.html'
    if request.method == 'POST':
        #form = UserCreationForm(request.POST)
        form = SignUpForm(request.POST)
        
        if form.is_valid():

            user = form.save(commit=False)
            user.is_active = False
            
            user.save()
            current_site = get_current_site(request)
            subject = 'Aktivieren Sie Ihren ML4ALL-Zugang'
            message = render_to_string('accounts/activmail.html', {
                'user': user,
                'domain': current_site.domain,
                'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                'token': account_activation_token.make_token(user),
            })


            user.email_user(subject, message)
            #return redirect('activ_sent')            
            context = {'msg': 'Um Ihren Zugang zu aktivieren klicken Sie bitte auf den Aktivierungslink, den wir Ihnen soeben per E-Mail zugeschickt haben.'}
            return render(request,template,context) 

    else:
        form = SignUpForm()
    return render(request, 'accounts/registration.html', {'form': form})

class Activate(View):
    template = 'accounts/info.html'
    def get(self, request, uidb64, token):        

        try:
            uid = force_text(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
        except(TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None

        if user is not None and account_activation_token.check_token(user, token):
            # activate user and login:            
            user.is_active = True
            user.user.email_confirmed = True
            user.save()
            login(request, user)
                        
            #return HttpResponse('Erfolgreich aktiviert.')
            context = {'msg':'Ihr Zugang ist nun aktiviert.'}

        else:
            #return HttpResponse('Activation link is invalid!')
            context = {'msg':'Der Aktivierungslink ist ungültig. Ein neuer Link wurde Ihnen soeben zugeschickt'}

            current_site = get_current_site(request)
            subject = 'Aktivieren Sie Ihren ML4ALL-Zugang'
            message = render_to_string('accounts/activmail.html', {
                'user': user,
                'domain': current_site.domain,
                'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                'token': account_activation_token.make_token(user),
            })
            user.email_user(subject, message)


        
        return render(request,self.template,context) 

            
############################################################################
#                           DASHBOARD
############################################################################
class DashboardView(LoginRequiredMixin, View): #, TemplateView):
    #model=file
    login_url = 'login'
    redirect_field_name = 'redirect_to'
    template = "accounts/dashboard.html"

    def get(self,request, *args, **kwargs):    
        context = {}
        login_user_id = self.request.user.pk
        context['files'] = file.objects.filter(user = login_user_id).order_by('-date_created')
        return render(request,self.template,context)

    def post(self, request, *args, **kwargs): 

        if 'path' in request.FILES:
            form = UploadForm(request.POST, request.FILES)             
            if form.is_valid():     
                user = User.objects.get(pk=request.user.pk)

                file_model_instance = file(path=request.FILES['path'], 
                                           user =  user 
                                           )
                file_model_instance.save()

                #clean dataset, overwrite original file
                try:
                    clean_dataset(file_model_instance) #path.name
                except:
                    ctx = {'errmsg':"""Beim Lesen der Datei ist ein Fehler aufgetreten.
                                    Vergewissern Sie sich dass es sich um eine gültige Datei vom Typ CSV, XLS oder XLSX handelt """ }
                    ctx['files'] = file.objects.filter(user = request.user.pk).order_by('-date_created')
                    return render(request,self.template_name,ctx)                        
           
                
                header,cols,types,minvals,maxvals,medvals,histlist,counts,rows = \
                            create_dataset_from_file(file_model_instance.path)

                hist_bytes = pickle.dumps(histlist)
                hist_base64 = base64.b64encode(hist_bytes)

                file_model_instance.header=header
                file_model_instance.cols=cols
                file_model_instance.types=types
                file_model_instance.minvals=minvals
                file_model_instance.maxvals=maxvals
                file_model_instance.medvals=medvals
                file_model_instance.histo=histlist            
                file_model_instance.counts=counts
                file_model_instance.rows=rows
                file_model_instance.histo = hist_base64
                file_model_instance.save()

                return redirect(reverse_lazy('dashboard'))
            else:
                ctx = {'errmsg':form.errors['path'] }
                ctx['files'] = file.objects.filter(user = request.user.pk).order_by('-date_created')
                return render(request,self.template_name,ctx)
        else: #delete
            login_user_id = self.request.user.pk
            file_id = int(request.POST['del'])            
            file_model_instance = file.objects.filter(user = login_user_id).get(pk=file_id)
            file_model_instance.delete()
            # file is deleted automatically 
            return redirect(reverse_lazy('dashboard'))

############################################################################
#                           DATASET
############################################################################
class DatasetsView(LoginRequiredMixin, View): 
    login_url = 'login'
    redirect_field_name = 'redirect_to'

    def get(self,request, *args, **kwargs):            
        login_user_id = self.request.user.pk
        context={'files':file.objects.filter(user = login_user_id)}
        return render(request,'accounts/datasets.html',context)

############################################################################
#                           DATASET DETAIL
############################################################################
class DatasetDetailView(LoginRequiredMixin, View): 
    login_url = 'login'
    redirect_field_name = 'redirect_to'
    template="accounts/datasets.html"


    def get(self,request,dataset_id):    
        
        login_user_id = self.request.user.pk
        files_obj = file.objects.get(id = dataset_id)
        cols=files_obj.cols
        header= files_obj.header.split(',')
        types = files_obj.types.split(',')
        counts = files_obj.counts.split(',') 
        minvals = files_obj.minvals.split(',')
        maxvals = files_obj.maxvals.split(',')
        medvals = files_obj.medvals.split(',')
        
        # pickle unload gibt arrays aus
        # muss die nachkommastellen trimmen
        hist_bytes = base64.b64decode(files_obj.histo)
        hist_list = pickle.loads(hist_bytes)
        hist_counts =[]
        hist_edges=[]

        secondtolastbinedge =[] #das wird für die korrekte darstellung der bins im histo benötigt
        lastbinedge =[]

        for idx,h in enumerate(hist_list):
            if len(h)>1:
                if types[idx]=='123':
                    hist_counts_str = ','.join('%.1f' % i for i in h[0])
                    hist_edges_str  = ','.join('%.1f' % i for i in h[1])
                    hist_counts.append(hist_counts_str)
                    hist_edges.append(hist_edges_str)                
                
                    secondtolastbinedge.append('%.1f' % (h[1][4])) #2nd to last bin-edge, 5 bins in total
                    lastbinedge.append('%.1f' % (h[1][5])) 

                    print("123")
                    print("counts:" + hist_counts_str)
                    print("edges:" + hist_edges_str)
                    print("\n\n")

                else:
                    hist_counts_str = ','.join('%d' % i for i in h[0])                    
                    hist_edges_str  = ','.join('"%s"' % i for i in h[1])                    
                    hist_counts.append(hist_counts_str)
                    hist_edges.append(hist_edges_str)                

                    print("ABC")
                    print("counts:" + hist_counts_str)
                    print("edges:" + hist_edges_str)
                    print("\n\n")

                    secondtolastbinedge.append(0.0)
                    lastbinedge.append(0)                    
            else:
                hist_counts.append('0')
                hist_edges.append('0')
                secondtolastbinedge.append(0.0)
                lastbinedge.append(0)

        detail = {
            'dataset_id':dataset_id,
            'columns':cols,
            'header':header,
            'types':types,
            'minvals':minvals,
            'maxvals':maxvals,
            'medvals':medvals,
            'counts':counts,
            'hist_counts':hist_counts,
            'hist_edges':hist_edges,
            'secondtolastbinedge':secondtolastbinedge,
            'lastbinedge':lastbinedge,
        }
        context={'files':file.objects.filter(user = login_user_id),
                 'detail':detail
                }

        return render(request,self.template,context)        




############################################################################
#                           MODEL LIST VIEW
############################################################################
class ModelListView(LoginRequiredMixin, View):
    login_url = 'login'
    redirect_field_name = 'redirect_to'
    template="accounts/model_listview.html" 

    def get(self,request):  
        context={}
        
        msg = request.session.get('msg',False)
        if msg:                         
            del(request.session['msg'])
            context['errmsg'] = msg            

        login_user_id = self.request.user.pk        
        headers = []
        files_obj = file.objects.filter(user = login_user_id)      
        mlmodels = MLmodel.objects.filter(user = login_user_id).order_by('-date_created')          

        # get data from query set by iterating through it
        for fobj in files_obj:
            headers.append(fobj.header.split(','))
        
        context['files'] = files_obj
        context['mlmodels'] = mlmodels 
        context['headers'] = headers

        return render(request,self.template,context)

    def post(self, request, *args, **kwargs): 
        
        #ACHTUNG: Besonderheit bei POST via AJAX:
        #Man sollte nach ajax-request kein redirect, render etc. nutzen, 
        #sondern jsonresponse

        if 'objective' in request.POST:
            objective = request.POST['objective'][0:]
            pk = int(request.POST['pk'][0:])
            modeltype = request.POST['modeltype'][0:]

            if modeltype == '-1':
                msg='Kein Modell gewählt.'
                request.session['msg']=msg
                return JsonResponse({ 'success': False })                

            # INPUT VALIDATION            
            try:
                dataset = file.objects.get(pk=pk)
                # -> z.B. titanic.csv
            except:
                msg='Datensatz nicht gewählt oder vorhanden.'
                request.session['msg']=msg
                #return redirect(request.path)
                return JsonResponse({ 'success': False })
            
            if objective == '-1':               
                msg='Keine Zielfunktion ausgewählt.'
                request.session['msg']=msg
                return JsonResponse({ 'success': False })
                #return redirect(request.path)                          
                
                
            try: 
                userinstance = user.objects.get(user=request.user.pk)
                if userinstance.running_jobs > 0:
                    result = {'success':False, 
                              'msg':'In der kostenfreien Nutzung ist die Zahl der laufenden Analysen auf 1 beschränkt.', 
                              'outfile':None}                
                else:
                    userinstance.running_jobs += 1
                    userinstance.save()

                    result = myml.myMLmodel('./data/' + str(dataset.path.name), 
                                         objective, 
                                         typeof=modeltype, 
                                         userinstance = userinstance)

                    userinstance.running_jobs -= 1
                    userinstance.save()                     
            except Exception as e: 
                msg = e
                request.session['msg']=msg
                userinstance.running_jobs -= 1
                userinstance.save()
                return JsonResponse({ 'success': False })

            if result['success'] != True:
                msg = result['msg']
                request.session['msg']=msg
                return JsonResponse({ 'success': False })


            if modeltype=='AN': # weil hier result-file pickle ist
                localfile = open(result['outfile'],'rb')     # wichtig, sonst decode error
            else: 
                localfile = open(result['outfile']) 
            djangofile = File(localfile)

            ml_model_instance = MLmodel(results=djangofile,
                                        user =  User.objects.get(pk=request.user.pk),
                                        model_type = modeltype,
                                        objective=objective,
                                        dataset=dataset,
            )
            ml_model_instance.save()                

            localfile.close()
            djangofile.close()
            os.remove(result['outfile'])
        else: 
            #delete result request                                                    
            login_user_id = self.request.user.pk
            pk = int(request.POST['del'])            
            ml_model_instance = MLmodel.objects.get(pk=pk)
            ml_model_instance.delete()            

        return redirect(reverse_lazy('model_listview'))

############################################################################
#                           MODEL RESULT VIEW
############################################################################
class ResultsView(LoginRequiredMixin, View):
    login_url = 'login'
    redirect_field_name = 'redirect_to'
    templates = { 
                    'FC':'accounts/decisiontree_classification.html',
                    'FR':'accounts/decisiontree_regression.html',
                    'LR':'accounts/logres.html',
                    'KC':'accounts/cluster.html',
                    'AN':'accounts/anomaly.html',
                    'AS':'accounts/assoziation.html',
                    'KO':'accounts/correlation.html',
                    'ARI':'accounts/arima.html',
                    'GBM':'accounts/gbm.html',
                    'SVC':'accounts/svc.html',
                    'SVR':'accounts/svr.html'
                }    

    def get(self,request,mlmodelid):        
        login_user_id = self.request.user.pk
        ml_obj = MLmodel.objects.get(id = mlmodelid)

        template = self.templates[ml_obj.model_type]

        path = ml_obj.results.url
        if ml_obj.model_type == 'AN': # besonders: result file ist pickle
            context = AnomalyContext(path)
        else:
            context = {'filepath':path }
        return render(request,template,context)

############################################################################
#                           AUX FUNCTIONS
############################################################################
def create_dataset_from_file(fname):    
    df = pd.read_pickle('./data/' + str(fname),compression='gzip')

    # dont't use datetime columns here
    df = df.select_dtypes(exclude=['datetime'])    
 

    header =','.join(i for i in df.columns)
    cols = len(df.columns)
    typedir = {'int64':'123', 'float64':'123','object':'ABC'}    
    types = ','.join(typedir[str(i)] for i in df.dtypes)
    counts = ','.join(str(df.count()[i]) for i in df.columns)
    rows = df.shape[0]

    minvals =list()
    maxvals=list()
    medvals=list()
    histolist = list()

    for colname in df.columns:
        if str(df[colname].dtype) != 'object':
            #print(df[colname].dtype)
            minvals.append(np.round(df[colname].min(),decimals=2))
            maxvals.append(np.round(df[colname].max(),decimals=2))
            medvals.append(np.round(df[colname].median(),decimals=2))

            hist,bin_edges = np.histogram(df[colname],bins=5)
            histolist.append((hist,bin_edges))
        else:
            minvals.append('-')
            maxvals.append('-')
            medvals.append('-')
            classes = df[colname].unique()

            histcounts = np.array(df[colname].value_counts().to_list())
            histolist.append((histcounts,classes))


    minvals_str =','.join(str(i) for i in minvals)
    maxvals_str =','.join(str(i) for i in maxvals)
    medvals_str =','.join(str(i) for i in medvals)



    return header,cols,types, minvals_str,maxvals_str, medvals_str, histolist, counts, rows


##############################################
#   CLEAN DATASET
##############################################        
def clean_dataset(file_model_instance):
    upper_thresh = 20
    lower_thresh = 10

    infilepath = './data/' + file_model_instance.path.name

    mime_type = magic.from_file(infilepath, mime=True)
    
    if mime_type == 'text/plain' or mime_type == 'application/csv':
        df=pd.read_csv(infilepath) #,delimiter=',')
    elif mime_type == 'application/vnd.oasis.opendocument.spreadsheet':
        df=pd.read_excel(infilepath,engine='odf')
    else: 
        df=pd.read_excel(infilepath)

    uselesscols=[]
    for i in df:
        if df[i].dtypes=='object':
        ########################
        # keep datetime columns 
        ########################
            guessed_format = _guess_datetime_format_for_array(df[i].to_numpy())
            if guessed_format != None: 
                df[i]=pd.to_datetime(df[i],format=guessed_format)
                print("Erkanntes Datetimeformat:%s" % (guessed_format))
                continue

            cnts=len(df[i].unique())
            if cnts > upper_thresh:
                uselesscols.append(i)

    df.drop(columns=uselesscols,inplace=True)  
    # NANs
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    df.loc[:, num_cols] = df.loc[:, num_cols].fillna(0)
    df.loc[:, cat_cols] = df.loc[:, cat_cols].fillna('Unbekannt')

    # drops row containing strange chars
    for i in df.columns:
        if df[i].dtypes == 'object':
            rows=df[i].str.contains("\\\\")
            df=df[rows==False]

    #convert types
    for col in df:
        #if df[col].dtypes == 'number': #!= 'object':
        if df[col].dtype.kind in 'biufc':
            if len(df[col].unique()) < lower_thresh:
                df[col] = df[col].astype('object')  

    df.to_pickle(infilepath, compression='gzip')




############################################################################
#                           Anomaly  Preproc
############################################################################
def AnomalyContext(infile):
    #load pickle 
    with open('./'+infile, 'rb') as f: # das './' ist wichtig, da hier keine url sondern
        top5list = pickle.load(f)       # syspath benötigt wird!

    featurenames = []
    scores = []
    featuresets = list()                         

    for idx in range(len(top5list)-1):
        i = top5list[idx]
        featname=i['feature']

        featurenames.append(featname)        
        scores.append('%.2f' % i['score'])

        tmplist=list()                    
        for n in i['top5']:            
            tmplist.append(n.tolist())

        featuresets.append(tmplist)

    headers=top5list[-1]


    context = {'featurenames':featurenames, 
                'scores':scores,
                'featuresets':featuresets,
                'headers':headers,
    }
    return context 