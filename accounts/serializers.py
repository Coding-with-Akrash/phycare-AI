from rest_framework import serializers
from django.contrib.auth import authenticate
from .models import User, Doctor, Patient, ChatLog, Appointment

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'role', 'is_verified']

class DoctorSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = Doctor
        fields = '__all__'

class PatientSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = Patient
        fields = '__all__'

class RegisterDoctorSerializer(serializers.Serializer):
    username = serializers.CharField()
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
    full_name = serializers.CharField()
    specialization = serializers.CharField()
    experience = serializers.IntegerField()
    clinic_location = serializers.CharField()
    available_hours = serializers.CharField()
    whatsapp_number = serializers.CharField(required=False)

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password'],
            role='doctor'
        )
        doctor = Doctor.objects.create(
            user=user,
            full_name=validated_data['full_name'],
            specialization=validated_data['specialization'],
            experience=validated_data['experience'],
            clinic_location=validated_data['clinic_location'],
            available_hours=validated_data['available_hours'],
            whatsapp_number=validated_data.get('whatsapp_number', ''),
        )
        return doctor

class RegisterPatientSerializer(serializers.Serializer):
    username = serializers.CharField()
    first_name = serializers.CharField(required=False)
    last_name = serializers.CharField(required=False)
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
    date_of_birth = serializers.DateField(required=False)
    age = serializers.IntegerField()
    gender = serializers.CharField()
    occupation = serializers.CharField()
    phone = serializers.CharField(required=False)
    address = serializers.CharField(required=False)
    city = serializers.CharField(required=False)
    state = serializers.CharField(required=False)
    emergency_contact = serializers.CharField(required=False)
    emergency_phone = serializers.CharField(required=False)

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data['username'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', ''),
            email=validated_data['email'],
            password=validated_data['password'],
            role='patient'
        )
        patient = Patient.objects.create(
            user=user,
            date_of_birth=validated_data.get('date_of_birth'),
            age=validated_data['age'],
            gender=validated_data['gender'],
            occupation=validated_data['occupation'],
            phone=validated_data.get('phone'),
            address=validated_data.get('address'),
            city=validated_data.get('city'),
            state=validated_data.get('state'),
            emergency_contact=validated_data.get('emergency_contact'),
            emergency_phone=validated_data.get('emergency_phone')
        )
        return patient

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)

    def validate(self, data):
        user = authenticate(**data)
        if user and user.is_active:
            return user
        raise serializers.ValidationError("Invalid credentials")

class ChatLogSerializer(serializers.ModelSerializer):
    facial_image = serializers.ImageField(required=False, write_only=True)

    class Meta:
        model = ChatLog
        fields = '__all__'

class AppointmentSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.user.username', read_only=True)
    doctor_name = serializers.CharField(source='doctor.full_name', read_only=True)

    class Meta:
        model = Appointment
        fields = '__all__'