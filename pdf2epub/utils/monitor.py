import os
import boto3
import argparse
from botocore.exceptions import NoCredentialsError, ClientError
import sys
import time
from utils.logging_config import configure_logging
from loguru import logger

# Configure logger (will be updated with verbose flag in main)
logger = None


def upload_directory(s3_client, local_folder, bucket_name, s3_folder):
    for root, dirs, files in os.walk(local_folder):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_path = os.path.join(s3_folder, relative_path)

            try:
                logger.info(f"Uploading {local_path} to {s3_path}")
                s3_client.upload_file(local_path, bucket_name, s3_path)
            except NoCredentialsError:
                logger.error("Credentials not available.")
                sys.exit(1)
            except ClientError as e:
                logger.error(f"Failed to upload {local_path} to {s3_path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")


def download_directory(s3_client, bucket_name, s3_folder, local_folder):
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        for obj in page.get('Contents', []):
            if not obj['Key'].endswith('/'):  # skip directories
                s3_file_path = obj['Key']
                local_file_path = os.path.join(local_folder, os.path.relpath(s3_file_path, s3_folder))
                local_file_dir = os.path.dirname(local_file_path)

                # Ensure the directory exists
                os.makedirs(local_file_dir, exist_ok=True)

                try:
                    logger.info(f"Downloading {s3_file_path} to {local_file_path}")
                    s3_client.download_file(bucket_name, s3_file_path, local_file_path)
                except Exception as e:
                    logger.error(f"Failed to download {s3_file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Upload or download files between a local directory and an S3 bucket.")
    parser.add_argument("local_folder", help="Local folder to upload from or download to")
    parser.add_argument("s3_folder", help="S3 folder to upload to or download from")
    parser.add_argument("access_key", help="S3 Access Key ID")
    parser.add_argument("secret_key", help="S3 Secret Access Key")
    parser.add_argument("endpoint_url", help="S3 Endpoint URL")
    parser.add_argument("bucket_name", help="S3 Bucket Name", default="bookepub")
    parser.add_argument("--final", action="store_true", help="Run upload once and then quit")
    parser.add_argument("--download", action="store_true", help="Download from S3 instead of uploading")
    parser.add_argument("--verbose", action="store_true", help="Display detailed information during execution")

    args = parser.parse_args()
    
    # Initialize logger with verbose flag
    global logger
    logger = configure_logging(verbose=args.verbose)
    
    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key,
        endpoint_url=args.endpoint_url
    )

    # Verify bucket existence
    try:
        s3_client.head_bucket(Bucket=args.bucket_name)
    except ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            logger.error(f"The bucket {args.bucket_name} does not exist.")
            sys.exit(1)
        else:
            logger.error(f"An error occurred: {e}")
            sys.exit(1)

    if args.download:
        download_directory(s3_client, args.bucket_name, args.s3_folder, args.local_folder)
        logger.info("Download complete. Exiting.")
    elif args.final:
        upload_directory(s3_client, args.local_folder, args.bucket_name, args.s3_folder)
        logger.info("Upload complete. Exiting.")
    else:
        while True:
            upload_directory(s3_client, args.local_folder, args.bucket_name, args.s3_folder)
            logger.info("Waiting 5 minutes before next sync...")
            time.sleep(300)  # Sleep for 5 minutes


if __name__ == "__main__":
    main()
