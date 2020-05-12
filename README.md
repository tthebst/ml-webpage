# ML WEBPAGE
On this webpage you can explore state-of-the-art deep learning models of different categories.

https://ai-demo.ch

### Architecture

To deploy the models I used a mix of different tools:
  - Webpage: Run on a cloud compute instance deployed with docker-compose
  - Classification models: All deployed in google cloud functions
  - Object detection models: Containerized model and environment and deployed on google cloud run.
  - Generative models: DCGAN and Progessive Gan are also deployed in containers on google cloud run. Biggan besides the webserver container on the host machine due to RAM constraints.
  - Language models: Transformer is currently not working due to RAM contraints. DeepSpeech is also deployed on clourun
  

All the source code is in this repository. 


### Note to self

Use following command to run docker compose on container optimized OS on GCP

```
docker run -v /var/run/docker.sock:/var/run/docker.sock -v "$PWD:/rootfs/$PWD" -w="/rootfs/$PWD" docker/compose:1.24.0 up
```
