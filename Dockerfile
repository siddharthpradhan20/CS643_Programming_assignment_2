FROM guoxiaojun2/spark-3.2.2-bin-hadoop3.2

WORKDIR /app

COPY . .

USER root

RUN apt-get update -y && apt-get install -y python3

RUN apt-get install -y python3-pip

RUN pip3 install pyspark findspark boto3 numpy pandas scikit-learn datetime

RUN chmod +x run_scripts.sh

USER ${SPARK_USER}

CMD spark-submit --master yarn CS643-WinePredictioncs643_program_assgn_2/WineTraining.py
CMD spark-submit --master yarn CS643-WinePredictioncs643_program_assgn_2/WineTesting.py
