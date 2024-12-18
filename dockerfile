# Use an official Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Install required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the app.py file and artifacts folder
COPY app.py .
COPY artifacts/model.pkl artifacts/preprocessor.pkl ./artifacts/
# COPY src/utils.py ./src/

# Expose the port on which FastAPI will run
EXPOSE 8080

# Set the command to run the FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]


#docker build -t mlproject_mathscore-fastapi .
#docker run -p 8080:8080 mlproject_mathscore-fastapi

#test data:
    # {
    #     "writing_score": 80.0,
    #     "reading_score": 85.0,
    #     "gender": "female",
    #     "race_ethnicity": "group B",
    #     "parental_level_of_education": "bachelor's degree",
    #     "lunch": "standard",
    #     "test_preparation_course": "completed"
    # }
    

#     {
#     "inputs": [
#         {
#             "writing_score": 80.0,
#             "reading_score": 85.0,
#             "gender": "female",
#             "race_ethnicity": "group B",
#             "parental_level_of_education": "bachelor's degree",
#             "lunch": "standard",
#             "test_preparation_course": "completed"
#         },
#         {
#             "writing_score": 72.0,
#             "reading_score": 78.0,
#             "gender": "male",
#             "race_ethnicity": "group A",
#             "parental_level_of_education": "high school",
#             "lunch": "free/reduced",
#             "test_preparation_course": "none"
#         }
#     ]
# }
