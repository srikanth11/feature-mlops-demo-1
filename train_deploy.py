import google
from google.cloud import storage
from google.cloud import aiplatform
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email
from python_http_client.exceptions import HTTPError
import os
import pandas as pd
import pytz
import json
import datetime

credentials_file = "tensile-nebula-406509-8fd0cc70c363.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_file

project_id = "tensile-nebula-406509"


def get_latest_dataset(bucket_name):    

    # Initialize the Google Cloud Storage client
    client = storage.Client(project=project_id)
    print(client)

    # Get the bucket
    bucket = client.get_bucket(bucket_name)
 
    # List all objects in the bucket
    blobs = bucket.list_blobs()
 
    ist = pytz.timezone('Asia/Kolkata')
 
    # Initialize variables to track the latest modified time and object
    latest_modified_time = None
    latest_modified_blob = None
    # Find the latest modified file
    for blob in blobs:
        # Convert last modified time to IST
        last_modified_ist = blob.updated.astimezone(ist)
       
        # Check if this is the latest modified file
        if latest_modified_time is None or last_modified_ist > latest_modified_time:
            latest_modified_time = last_modified_ist
            latest_modified_blob = blob
 
    if latest_modified_blob:
      formatted_time = latest_modified_time.strftime("%Y-%m-%d %I:%M:%S %p")
      return {
          "file_name": latest_modified_blob.name,
          "size": latest_modified_blob.size,
          "last_modified_ist": formatted_time
      }
    else:
      return None
    
def load_meta_data():
    with open('./meta_data.json', 'r') as f:
        meta_data = json.load(f)
    return meta_data

def email(_from_email, _to_emails, mail_api_key, message_data):
    api_key = mail_api_key
    sg = SendGridAPIClient(api_key)
    html_content = f"<h3>Auto ML Workshop model status</h3> \
                    <table border='1' style='border-collapse:collapse'> \
                        <tr> \
                            <td rowspan='2'> File Upload </td> \
                            <td> Status </td> \
                            <td> {message_data.get('file_upload').get('status')} </td> \
                        </tr> \
                        <tr> \
                            <td> Message </td> \
                            <td> {message_data.get('file_upload').get('message')} </td> \
                        </tr> \
                        <tr> \
                            <td rowspan='2'> Dataset </td> \
                            <td> Status </td> \
                            <td> {message_data.get('dataset').get('status')} </td> \
                        </tr> \
                        <tr> \
                            <td> Message </td> \
                            <td> {message_data.get('dataset').get('message')} </td> \
                        </tr> \
                        <tr> \
                            <td rowspan='2'> Training </td> \
                            <td> Status </td> \
                            <td> {message_data.get('training_job').get('status')} </td> \
                        </tr> \
                        <tr> \
                            <td> Message </td> \
                            <td> {message_data.get('training_job').get('message')} </td> \
                        </tr> \
                    </table>"
    # create a message to be sent
    message = Mail(
        from_email=_from_email,
        to_emails=_to_emails,
        subject="Test Mail MLOPS Workshop",
        html_content=html_content
        )
    # message.add_bcc("srikanth6835@gmail.com") 
    try:
        response = sg.send(message)
        return response
 
    except HTTPError as e:
        return e

def get_latest_dataset(bucket_name):
   
    # Initialize the Google Cloud Storage client
    client = storage.Client()
    # Get the bucket
    bucket = client.get_bucket(bucket_name)
 
    # List all objects in the bucket
    blobs = bucket.list_blobs()
 
    ist = pytz.timezone('Asia/Kolkata')
 
    # Initialize variables to track the latest modified time and object
    latest_modified_time = None
    latest_modified_blob = None
    # Find the latest modified file
    for blob in blobs:
        # Convert last modified time to IST
        last_modified_ist = blob.updated.astimezone(ist)
       
        # Check if this is the latest modified file
        if latest_modified_time is None or last_modified_ist > latest_modified_time:
            latest_modified_time = last_modified_ist
            latest_modified_blob = blob
 
    if latest_modified_blob:
      formatted_time = latest_modified_time.strftime("%Y-%m-%d %I:%M:%S %p")
      return {
          "file_name": latest_modified_blob.name,
          "size": latest_modified_blob.size,
          "last_modified_ist": formatted_time
      }
    else:
      return None

def datetime_to_str():
    ist = pytz.timezone('Asia/Kolkata')
    current_datetime = datetime.datetime.now(ist)
    formatted_date = current_datetime.strftime("%d-%m-%y")
    formatted_time = current_datetime.strftime("%H:%M")
    return f"{formatted_date}_{formatted_time}"

def get_dataset_id(file_name):
    try:
        time_stamp = datetime_to_str()
        file_display_name = f"dataset_{time_stamp}"
        my_dataset = aiplatform.TabularDataset.create(
            display_name=file_display_name, gcs_source=[file_name])
        dataset_id = my_dataset.name.split('/')[-1]
        return dataset_id
    except Exception as e:
        print(e)
        return e


def training_job(display_name, opt, dataset, target_column, des, model_name):
    try:
        job = aiplatform.AutoMLTabularTrainingJob(
            display_name=display_name,
            optimization_prediction_type=opt
        )
        model = job.run(
            dataset=dataset,
            target_column=target_column,
            training_fraction_split=0.8,
            validation_fraction_split=0.1,
            test_fraction_split=0.1,
            model_display_name = model_name,
            disable_early_stopping=des,
            model_version_description="This is the next version of the model"
        )

        return model, "successful"
    except Exception as e:
        error = f"Error during training job: {e}"
        return error, "error"

    return model

def hello_gcs1(project_id, region, endpoint_id, output_buk, opt, model_display_name, m_type, from_address, to_address, m_api_key):
    
    email_dict = {
        "file_upload" : {"status": None, "message": None},
        "dataset": {"status": None, "message": None},
        "training_job": {"status": None, "message": None},
        "model_deploy": {"status": None, "message": None},
        "expose_endpoint": {"status": None, "message": None},
        "endpoint_url": {"status": None, "message": None}
    }
    
    PROJECT_ID = project_id
    REGION = region

    endpoint_id = endpoint_id
    bucket_name = output_buk

    latest_file_info = get_latest_dataset(bucket_name)
   
    # if latest_file_info['file_name'] != None:
    #     email_dict['file_upload']['status'] = "Successs"
    #     email_dict['file_upload']['message'] = f"File uploaded to the bucket {output_buk} with the file name {latest_file_info['file_name']}"
    #     mail_obj = email(from_address, to_address, m_api_key, email_dict)
 
    file_name = f"gs://{bucket_name}/{latest_file_info['file_name']}"

    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    dataset_id = get_dataset_id(file_name)
    dataset = aiplatform.TabularDataset(dataset_id)

    print('Dataset ID: \t', dataset_id)

    # if dataset_id != None:
    #     email_dict['dataset']['status'] = "Success"
    #     email_dict['dataset']['message'] = f"Dataset created successfully with the Dataset ID {dataset_id}"
    #     mail_obj = email(from_address, to_address, m_api_key, email_dict)
 
    model, status = training_job("mlops-training-pipeline", "classification", dataset, "salary", True, "mlops-model-1")

    print(status)
    # if status == "successful":
    #     email_dict['training_job']['status'] = "Successs"
    #     email_dict['training_job']['message'] = f"Training Job completed successfully"
    #     mail_obj = email(from_address, to_address, m_api_key, email_dict)
    # elif status == "error":
    #     email_dict['training_job']['status'] = "Failure"
    #     email_dict['training_job']['message'] = f"Training job failed, please check logs"
    #     mail_obj = email(from_address, to_address, m_api_key, email_dict)

if __name__ == "__main__":
    meta_data = load_meta_data()

    project_id = meta_data.get('project_id')
    region = meta_data.get('region')
    endpoint_id = meta_data.get('endpoint_id')
    output_buk = meta_data.get('bucket_name')
    opt = meta_data.get('optimization_prediction_type')
    model_display_name = meta_data.get('model_display_name')
    m_type = meta_data.get('machine_type')
    from_email = meta_data.get('from_email')
    to_emails = meta_data.get('to_emails')
    mail_api_key = meta_data.get('mail_api_key')
 
    model_deployment_data = hello_gcs1(project_id, region, endpoint_id, output_buk, opt, model_display_name, m_type, from_email, to_emails, mail_api_key)
    print('Model Deployment Data: \t', model_deployment_data)





