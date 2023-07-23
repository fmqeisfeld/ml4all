from django import forms
# from django.forms import ModelForm
from .models import file, user



# for custom user registration form
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

# django simple captcha
from captcha.fields import CaptchaField

# overwrite password reset
from django.contrib.auth.forms import PasswordResetForm
# overwrite password change
from django.contrib.auth.forms import SetPasswordForm


# das brauche ich um die auto-generierten labels von crispy forms zu entfernen
from crispy_forms.helper import FormHelper

class UploadForm(forms.ModelForm):
    class Meta:
        model = file
        fields=['path']
            


class SignUpForm(UserCreationForm):
    email = forms.EmailField(max_length=254, help_text='Erforderlich. Bitte gültige E-Mail angeben.')
    captcha = CaptchaField(label='Sicherheitsprüfung:',initial='Bitte abgebildete Zeichen eingeben',show_hidden_initial=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2', 'captcha')        
          
class UserPasswordResetForm(PasswordResetForm):
    def __init__(self, *args, **kwargs):
        super(UserPasswordResetForm, self).__init__(*args, **kwargs)

    email = forms.EmailField(label='', widget=forms.EmailInput(attrs={
        'class': 'your class',
        'placeholder': 'Bitte E-Mail angeben',
        'type': 'email',
        'name': 'email'
        }))

    captcha = CaptchaField(label='Sicherheitsprüfung:',)

    def clean_email(self):
        email = self.cleaned_data['email']
        if not User.objects.filter(email__iexact=email, is_active=True).exists():
            msg = ("Es existiert kein Benutzer mit der angegebenen E-Mail.")
            self.add_error('email', msg)
        return email

    class Meta:
        model = User
        fields = ('email', 'captcha')   

# das selbe für password change
class UserSetPasswordForm(SetPasswordForm):
    captcha = CaptchaField(label='Sicherheitsprüfung:')

    def __init__(self, *args, **kwargs):
        super(UserSetPasswordForm, self).__init__(*args, **kwargs)
    

    class Meta:
        model = User
        fields = ('new_password1','new_password2','captcha')   
    
#################    
# CONFIG
################
class ConfigForm(forms.ModelForm):    

    def __init__(self, *args, **kwargs):
        super(ConfigForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_show_labels = False 


        try:            
            for key in self.fields:
                self.fields[key].initial = getattr(self.instance,key)
        except:
            pass
        

    #####################
    # FORM VALIDATION
    #####################    

    ###################
    # LOGREG
    #################
    def clean_logreg_C(self):
        C = self.cleaned_data['logreg_C']
        if C < 0:
            raise forms.ValidationError("Der Regularisierungsparameter muss > 0 sein.")
        return C
    
    def clean_logreg_maxiter(self):
        foo = self.cleaned_data['logreg_maxiter']
        if foo == -1:
            raise forms.ValidationError("In der kostenfreien Nutzung beschränkt sich die Zahl der maximalen Iterationen auf 100000.")
        return foo

    def clean_logreg_optimloops(self):
        foo = self.cleaned_data['logreg_optimloops']
        if foo == -1:
            raise forms.ValidationError("In der kostenfreien Nutzung ist keine Hyperparameter-Optimierung möglich.")
        return foo

    ###################
    # SVC
    #################
    def clean_svc_C(self):
        C = self.cleaned_data['svc_C']
        if C < 0:
            raise forms.ValidationError("Der Regularisierungsparameter muss > 0 sein.")
        return C
    
    def clean_svc_maxiter(self):
        foo = self.cleaned_data['svc_maxiter']
        if foo == -1:
            raise forms.ValidationError("In der kostenfreien Nutzung beschränkt sich die Zahl der maximalen Iterationen auf 100000.")
        return foo

    def clean_svc_optimloops(self):
        foo = self.cleaned_data['svc_optimloops']
        if foo == -1:
            raise forms.ValidationError("In der kostenfreien Nutzung ist keine Hyperparameter-Optimierung möglich.")        
        return foo

    def clean_svc_kernel(self):
        kernel = self.cleaned_data['svc_kernel']
        if kernel == 'rbf':
            raise forms.ValidationError("In der kostenfreien Nutzung ist kein RBF-Kernel möglich.")           
        return kernel

    ###################
    # SVR
    #################
    def clean_svr_C(self):
        C = self.cleaned_data['svr_C']
        if C < 0:
            raise forms.ValidationError("Der Regularisierungsparameter muss > 0 sein.")
        return C
    
    def clean_svr_maxiter(self):
        foo = self.cleaned_data['svr_maxiter']
        if foo == -1:
            raise forms.ValidationError("In der kostenfreien Nutzung beschränkt sich die Zahl der maximalen Iterationen auf 100000.")
        return foo

    def clean_svr_optimloops(self):
        foo = self.cleaned_data['svr_optimloops']
        if foo == -1:
            raise forms.ValidationError("In der kostenfreien Nutzung ist keine Hyperparameter-Optimierung möglich.")        
        return foo

    def clean_svr_kernel(self):
        kernel = self.cleaned_data['svr_kernel']
        if kernel == 'rbf':
            raise forms.ValidationError("In der kostenfreien Nutzung ist kein RBF-Kernel möglich.")                   
        return kernel

    ###################
    # ARIMA
    #################
    def clean_armima_maxpq(self):
        foo = self.cleaned['arima_maxpq']
        if foo == -1:
            raise forms.ValidationError("In der kostenfreien Nutzung ist die maximale Ordnung der (P,Q)-Parameter auf 5 beschränkt.")
        return foo

    def clean_arima_optimloops(self):
        foo = self.cleaned_data['arima_optimloops']
        if foo == -1:
            raise forms.ValidationError("In der kostenfreien Nutzung ist die Zahl der Optimierungsiterationen auf 2 beschränkt.")                          
        return foo

    ###################
    # LGBM
    #################
    def clean_lgbm_optimloops(self):
        foo = self.cleaned_data['lgbm_optimloops']
        if foo == -1:
            raise forms.ValidationError("In der kostenfreien Nutzung ist die Zahl der Optimierungsiterationen auf 5 beschränkt.")            
        return foo

    class Meta:
        model = user
        fields=['logreg_maxiter','logreg_testsize','logreg_C','logreg_optimloops',
                'svc_maxiter','svc_testsize','svc_C','svc_optimloops','svc_degree','svc_kernel',
                'svr_maxiter','svr_testsize','svr_C','svr_optimloops','svr_degree','svr_kernel', 
                'arima_maxpq','arima_testsize','arima_forecast','arima_optimloops',
                'lgbm_testsize','lgbm_forecast','lgbm_optimloops'
        ]

        # min value for C
        widgets = {'logreg_C':forms.TextInput(attrs={'min':1.0}),
                   'svc_C':forms.TextInput(attrs={'min':1.0}),
                   'svr_C':forms.TextInput(attrs={'min':1.0}) 
        }        
