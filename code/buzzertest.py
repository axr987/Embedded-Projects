import time
import RPi.GPIO as GPIO

note_interval = 2 ** (1/12)
note_start = 523  # C5 frequency in Hz

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)  # Alarm output pin
buzzer_pwm = GPIO.PWM(18, note_start)  # Buzzer on GPIO 18 at 1kHz
buzzer_pwm.start(0)  # Start with buzzer off

def cleanup_GPIO():
    buzzer_pwm.stop()
    GPIO.cleanup()

while True:
    buzzer_pwm.ChangeDutyCycle(50)  # 50% duty cycle for audible tone
    buzzer_pwm.ChangeFrequency(note_start)
    time.sleep(0.05)  
    buzzer_pwm.ChangeFrequency(note_interval * note_start)
    time.sleep(0.05)  
    buzzer_pwm.ChangeFrequency(note_interval ** 11 * note_start)
    time.sleep(0.10)  
    buzzer_pwm.ChangeDutyCycle(0)  
    time.sleep(0.05)  
    buzzer_pwm.ChangeDutyCycle(50)  
    buzzer_pwm.ChangeFrequency(note_start)
    time.sleep(0.05)  
    buzzer_pwm.ChangeFrequency(note_interval * note_start)
    time.sleep(0.05)  
    buzzer_pwm.ChangeFrequency(note_interval ** 11 * note_start)
    time.sleep(0.10)
    buzzer_pwm.ChangeDutyCycle(0)
    time.sleep(0.05)

cleanup_GPIO()