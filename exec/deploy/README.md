외부 서비스와 DB 덤프파일은 이용할 것이 없어 생략했습니다.

# 1

**사용 프레임워크/ 라이브러리**

- vue cli 2.0
- typescript
- vuetify
- vue router
- axios

### 프론트엔드

- 배포
    
     사이트가 간단하기 때문에 npm run build를 사용하여 빌드된 정적 파일을 Nginx에 설정하여 배포했습니다.
    
    SSL, HTTPS 필수! 카메라 접근은 HTTP는 불가능합니다.
    
    Nginix conf파일
    
    ```jsx
    server {
        listen 443 ssl;
        server_name j6b201.p.ssafy.io;
        ssl_certificate /etc/letsencrypt/live/j6b201.p.ssafy.io/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/j6b201.p.ssafy.io/privkey.pem;
        root    /home/ubuntu/S06P22B201/frontend/dist;
        index    index.html index.htm;
        location @rewrites {
            rewrite ^(.+)$ /index.html last;
        }
        location / {
            try_files $uri $uri/ @rewrites;
        }
    }
    ```
    


### 백엔드

![스크린샷 2022-04-07 오후 5.25.15.png](image/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2022-04-07_%EC%98%A4%ED%9B%84_5.25.15.png)

![스크린샷 2022-04-07 오후 5.25.19.png](image/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2022-04-07_%EC%98%A4%ED%9B%84_5.25.19.png)

![스크린샷 2022-04-07 오후 5.25.25.png](image/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2022-04-07_%EC%98%A4%ED%9B%84_5.25.25.png)

![스크린샷 2022-04-07 오후 5.25.30.png](image/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2022-04-07_%EC%98%A4%ED%9B%84_5.25.30.png)

![스크린샷 2022-04-07 오후 5.25.36.png](image/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2022-04-07_%EC%98%A4%ED%9B%84_5.25.36.png)