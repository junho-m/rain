About
-----

딥러닝으로 강수량을 예측해본다. 

If you use the code or find it helpful, please cite the following paper:
```
	@inproceedings{juno1moon@gmail.com,
	    title={Deep learning for precipitation RainFall},
	    author={문준호},
	    booktitle={딥러닝 강수량 예측},
	    year={2020}
	}
```

Installation
------------
   
    - 참고문서 : /rain/doc/파이선+텐서플로+이클립스+git.docx
    
os: windows10

1.아나콘다 설지

    ① 다운로드후 설지     
        https://www.anaconda.com/distribution/#download-section
        
    ② PowerShell 실행(관리자모드)    
    
        A. conda update 수행        
           i. conda update conda            
          ii. conda update anaconda            
         iii. conda update python            
          iv. conda update –all
            
        B. conda 가상환경 만들기        
           i. conda info –envs            
          ii. conda create --name rain python=3.7
            
2.Git 생성

    ①  https://github.com/
    
    ②  프로젝트 생성     
        예제) https://github.com/junho-m/rain
        
3.이클립스 생성

    ① https://www.eclipse.org/downloads/packages/installer

4.이클립스 설정 

    ① Help>Eclipse Marketplace에서 PyDev 설치    
    ② PyDev,Git연결

5.패키지  설치

    - 참고문서 : /rain/doc/파이선+텐서플로+이클립스+git.docx
    
    - 주의 사항 : 
        
        Git에서 예제를 다운받으면 설정이 이미 되어 있습니다. 
    
        이클립스에서 Django 프로젝트 생성하면 장고환경이 오류나기도 해서
        
        작업을 Powershell 에서 하실것을 권유해 드립니다. 
        


DB 설정
------------

PowerShell에서 작업을 진행한다. 

	conda activate rain
	
	django-admin startproject rain 
	
	cd C:\WORK\rain #프로젝트위치로 이동 
	
	python manage.py migrate
	
	python manage.py createsuperuser
	
	
	이클립스 Market Place에서 
	
	DBeaver 설치 ( sqlite 관리를 하기 위해 설치함)


NVIDA GPU설정(이부분은 안하셔도 됩니다.) 
------------

1.최신 드라이버 설치 ( 드라이버 중에 CUDA 10.1 버전 호환되는 최종버전은 아래와 같다)

    https://chan-lab.tistory.com/13?category=805685
    
	AnacondaAnaconda3-2019.03-Windows-x86_64
	vs_community_2017_15.9.14
	cuda_10.0.130_411.31_win10
	cudnn-10.0-windows10-x64-v7.6.0.64

2.Visual Studio 2017 설치하기

    https://docs.microsoft.com/ko-kr/visualstudio/releasenotes/vs2017-relnotes
    
3.Cuda Toolkit 10.0 설치하기

    https://developer.nvidia.com/cuda-toolkit-archive
    
4.CuDNN 파일 다운

    https://developer.nvidia.com/rdp/cudnn-archive


참고자료
------------

1.FORECA
	
	https://www.foreca.com/Korea/Seoul?map=rain
	
2.DACON ( 강수량 산출 대회 )

	https://dacon.io/competitions/official/235591/data/

3.참고소스 

	https://github.com/sxjscience/HKO-7.git	
