# docker-Florence-2-large-PromptGen-v2.0
basic http api for MiaoshouAI/Florence-2-large-PromptGen-v2.0

## Usage

Build image:
```
docker build -t florence2-promptgen https://github.com/veka-server/docker-Florence-2-large-PromptGen-v2.0.git#main
```

Basic example:
```
docker run \
--net reseaux_sans_internet \
--restart unless-stopped \
-p 5000:5000  \
--name florence2-promptgen \
-d florence2-promptgen;
```
