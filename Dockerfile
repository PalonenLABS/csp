FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt into the container
COPY . .

# Install pipenv and dependencies
RUN pip install pipenv && pipenv install --system --deploy

#RUN pip3 install --user pipenv
#RUN pipenv install
#RUN pipenv shell
# Install dependencies
# RUN pip install 
##--no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
#COPY . .

# Run the application
#CMD ["python", "csp/stock_cutter_1d.py"]

