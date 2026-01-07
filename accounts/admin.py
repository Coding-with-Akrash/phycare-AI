from django.contrib import admin
from django.utils import timezone
from .models import User, Doctor, Patient, MoodEntry, ChatLog, Conversation, MentalHealthQuiz, Appointment

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ['username', 'email', 'role', 'is_verified', 'is_active']
    list_filter = ['role', 'is_verified', 'is_active']
    search_fields = ['username', 'email']

@admin.register(Doctor)
class DoctorAdmin(admin.ModelAdmin):
    list_display = ['user', 'full_name', 'specialization', 'approved_by_admin', 'status']
    list_filter = ['approved_by_admin', 'status', 'specialization']
    search_fields = ['user__username', 'full_name', 'specialization']
    readonly_fields = ['user']

    actions = ['approve_doctors', 'reject_doctors']

    def approve_doctors(self, request, queryset):
        queryset.update(approved_by_admin=True, status='approved')
        self.message_user(request, f"{queryset.count()} doctors approved.")
    approve_doctors.short_description = "Approve selected doctors"

    def reject_doctors(self, request, queryset):
        queryset.update(approved_by_admin=False, status='rejected')
        self.message_user(request, f"{queryset.count()} doctors rejected.")
    reject_doctors.short_description = "Reject selected doctors"

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ['user', 'age', 'gender', 'occupation']
    list_filter = ['gender', 'occupation']
    search_fields = ['user__username', 'occupation']

@admin.register(MoodEntry)
class MoodEntryAdmin(admin.ModelAdmin):
    list_display = ['user', 'mood', 'created_at']
    list_filter = ['mood', 'created_at']
    search_fields = ['user__username', 'notes']
    readonly_fields = ['created_at']

@admin.register(ChatLog)
class ChatLogAdmin(admin.ModelAdmin):
    list_display = ['conversation', 'message', 'timestamp']
    list_filter = ['timestamp']
    search_fields = ['conversation__user__username', 'message']
    readonly_fields = ['timestamp']

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ['user', 'overall_emotion', 'overall_severity', 'is_active', 'started_at']
    list_filter = ['started_at']
    search_fields = ['user__username']
    readonly_fields = ['started_at', 'ended_at']

@admin.register(MentalHealthQuiz)
class MentalHealthQuizAdmin(admin.ModelAdmin):
    list_display = ['user', 'depression_level', 'anxiety_level', 'total_score', 'created_at']
    list_filter = ['depression_level', 'anxiety_level', 'created_at']
    search_fields = ['user__username']
    readonly_fields = ['created_at', 'answers', 'recommendations']

@admin.register(Appointment)
class AppointmentAdmin(admin.ModelAdmin):
    list_display = ['patient', 'doctor', 'date', 'time', 'status']
    list_filter = ['status', 'date']
    search_fields = ['patient__user__username', 'doctor__full_name']
