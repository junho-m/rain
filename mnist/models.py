from django.db import models

# Create your models here.

'''
컬럼명    컬럼설명
채널0    밝기 온도 (단위: K, 10.65GHz~89.0GHz)
채널1    밝기 온도 (단위: K, 10.65GHz~89.0GHz)
채널2    밝기 온도 (단위: K, 10.65GHz~89.0GHz)
채널3    밝기 온도 (단위: K, 10.65GHz~89.0GHz)
채널4    밝기 온도 (단위: K, 10.65GHz~89.0GHz)
채널5    밝기 온도 (단위: K, 10.65GHz~89.0GHz)
채널6    밝기 온도 (단위: K, 10.65GHz~89.0GHz)
채널7    밝기 온도 (단위: K, 10.65GHz~89.0GHz)
채널8    밝기 온도 (단위: K, 10.65GHz~89.0GHz)
채널9    지표 타입 (앞자리 0: Ocean, 앞자리 1: Land, 앞자리 2: Coastal, 앞자리 3: Inland Water)
채널10    GMI 경도
채널11    GMI 위도
채널12    DPR 경도
채널13    DPR 위도
채널14    강수량 (mm/h, 결측치는 -9999.xxx 형태의 float 값으로 표기)
'''


class Mnist(models.Model):
    fileName  = models.CharField(max_length=100)
    ch0 = models.CharField(max_length=100)
    ch1 = models.CharField(max_length=100)
    ch2 = models.CharField(max_length=100)
    ch3 = models.CharField(max_length=100)
    ch4 = models.CharField(max_length=100)
    ch5 = models.CharField(max_length=100)
    ch6 = models.CharField(max_length=100)
    ch7 = models.CharField(max_length=100)
    ch8 = models.CharField(max_length=100)
    ch9 = models.CharField(max_length=100)
    ch10 = models.CharField(max_length=100)
    ch11 = models.CharField(max_length=100)
    ch12 = models.CharField(max_length=100)
    ch13 = models.CharField(max_length=100)
    ch14 = models.CharField(max_length=100)
    
    pub_date = models.DateTimeField('date published')
    
    
    def __str__(self):
        return self.fileName    
    
    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)
