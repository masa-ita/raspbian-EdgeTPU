FROM balenalib/raspberrypi3-debian:stretch

RUN [ "cross-build-start" ]

#labeling
LABEL mantainer="Masatoshi Itagaki <itagakim@gmail.com>" \
    org.label-schema.build-date=$BUILD_DATE \
    org.label-schema.name="raspbian-edgetpu" \
    org.label-schema.description="Docker running Raspbian including Coral Edge-TPU libraries Derived from https://github.com/lemariva/raspbian-EdgeTPU" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/masa-ita/raspbian-EdgeTPU" \
    org.label-schema.vendor="Masatoshi Itagaki" \
    org.label-schema.version=$VERSION \
    org.label-schema.schema-version="1.0"

ENV READTHEDOCS True
ENV CONFIG_PATH="/root/.jupyter/jupyter_notebook_config.py"

#do installation
RUN apt-get update \
	&& apt-get install -y --no-install-recommends openssh-server \
#do users
    && echo 'root:root' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
    && mkdir /var/run/sshd

#install libraries for camera
RUN apt-get install -y --no-install-recommends build-essential wget feh pkg-config libjpeg-dev zlib1g-dev \
    libraspberrypi0 libraspberrypi-dev libraspberrypi-doc libraspberrypi-bin libfreetype6-dev libxml2 libopenjp2-7 \
    libatlas-base-dev libjasper-dev libqtgui4 libqt4-test \
    python3-dev python3-pip python3-setuptools python3-wheel python3-numpy python3-pil python3-matplotlib python3-zmq

#nodejs for notebooks
RUN curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash - \
    && apt-get install -y nodejs

#python libraries
RUN python3 -m pip install supervisor \
    && python3 -m pip install picamera python-periphery \
    && python3 -m pip install jupyter cython jupyterlab ipywebrtc opencv-python \
	&& python3 -m pip install google-auth oauthlib imutils

#jupyter packages
# RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager \
#     && jupyter labextension install jupyter-webrtc \
#     && jupyter nbextension enable --py widgetsnbextension

#install live camera libraries
RUN apt-get install libgstreamer1.0-0 gstreamer1.0-tools \ 
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \ 
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly v4l-utils

#installing edge-tpu library
WORKDIR /opt

#downloading library file 
RUN wget https://dl.google.com/coral/edgetpu_api/edgetpu_api_latest.tar.gz -O edgetpu_api.tar.gz --trust-server-names \
    && tar xzf edgetpu_api.tar.gz \
    && rm edgetpu_api.tar.gz

#SSL for jupyter    
RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem -subj '/O=LeMaRiva|tech/C=DE'

#trick platform recognizer 
COPY "./conf/install.sh" /opt/edgetpu_api/install.sh
#installing library
RUN cd /opt/edgetpu_api/ \
	&& chmod +x install.sh \
    && bash install.sh -y

#copy files
RUN mkdir /notebooks
COPY "./examples/*" /notebooks/
COPY "./conf/jupyter_notebook_config.py" ${CONFIG_PATH}

#loading pretrained models
WORKDIR /notebooks
RUN wget -P test_data/ https://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
    && wget -P test_data/ http://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/imagenet_labels.txt \
    && tar xvf examples_edgetpu.tar.xz \
    && rm examples_edgetpu.tar.xz 

#copy supervisord files
COPY "./conf/supervisord.conf" /etc/supervisor/conf.d/supervisord.conf
RUN mkdir /var/log/supervisord/

#supervisord
CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

#SSH Port
EXPOSE 22 8888 8080 8000

#set stop signal
STOPSIGNAL SIGTERM

#stop processing ARM emulation (comment out next line if built on Raspberry)
RUN [ "cross-build-end" ]
