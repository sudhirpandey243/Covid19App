import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Age':76, 'Gender':2, 'Do you smoke':5, 'Do you have any of these low impact symptoms?(Headache, Sore Throat, Chest Pain, Loss of Taste or Smell, Vomiting)':10, 'Do you have any of these high impact symptoms? (Fever, Cough, Shortness of Breath, Muscle Pain)':20, 'Do you have any of these health conditions? (Heart Disease, Asthma, Kidney Disease, Diabetes, Cancer, Pregnancy)':100, 'Are You Following These Guidelines Consistently(Frequent Hand wash with soap for at least 20 Seconds, Wearing Mask , Social Distancing, Self Quarantine, if unwell )':0, 'Have You Had Any of These Exposures in Last 4 Days(Travelled to Containment Zone, Living in Containment Zone, Face to Face Contact with a Confirmed Case of COVID19 for more then 15':1 })

print(r.json())