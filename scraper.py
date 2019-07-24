p=1
my_list=[]
import requests
from bs4 import BeautifulSoup

html_content=requests.get("https://karki23.github.io/Weather-Data/Albury.html").text
soup = BeautifulSoup(html_content, "lxml")

header=soup.find_all('th')
h_data=[i.text for i in header]
my_list.append(h_data)



def scrape(a):
    
   import requests
   from bs4 import BeautifulSoup

   link =a
   html_content=requests.get(link).text
   soup_1 = BeautifulSoup(html_content, "lxml")
   
   my_toup=[]

   stat_table= soup_1.find_all('table')
   stat_table=stat_table[0]

   for row in stat_table.find_all('tr'):
       for cell in row.find_all('td'):
         my_toup.append(cell.text)
       my_list.append(my_toup)
       my_toup=[]
       
       
       
lists=["https://karki23.github.io/Weather-Data/Albury.html","https://karki23.github.io/Weather-Data/BadgerysCreek.html","https://karki23.github.io/Weather-Data/Cobar.html","https://karki23.github.io/Weather-Data/CoffsHarbour.html","https://karki23.github.io/Weather-Data/Moree.html","https://karki23.github.io/Weather-Data/Newcastle.html","https://karki23.github.io/Weather-Data/NorahHead.html","https://karki23.github.io/Weather-Data/NorfolkIsland.html","https://karki23.github.io/Weather-Data/Penrith.html","https://karki23.github.io/Weather-Data/Richmond.html","https://karki23.github.io/Weather-Data/Sydney.html","https://karki23.github.io/Weather-Data/SydneyAirport.html","https://karki23.github.io/Weather-Data/WaggaWagga.html","https://karki23.github.io/Weather-Data/Williamtown.html","https://karki23.github.io/Weather-Data/Wollongong.html","https://karki23.github.io/Weather-Data/Canberra.html","https://karki23.github.io/Weather-Data/Tuggeranong.html","https://karki23.github.io/Weather-Data/MountGinini.html","https://karki23.github.io/Weather-Data/Ballarat.html","https://karki23.github.io/Weather-Data/Bendigo.html","https://karki23.github.io/Weather-Data/Sale.html","https://karki23.github.io/Weather-Data/MelbourneAirport.html","https://karki23.github.io/Weather-Data/Melbourne.html","https://karki23.github.io/Weather-Data/Mildura.html","https://karki23.github.io/Weather-Data/Nhil.html","https://karki23.github.io/Weather-Data/Portland.html","https://karki23.github.io/Weather-Data/Watsonia.html","https://karki23.github.io/Weather-Data/Dartmoor.html","https://karki23.github.io/Weather-Data/Brisbane.html","https://karki23.github.io/Weather-Data/Cairns.html","https://karki23.github.io/Weather-Data/GoldCoast.html","https://karki23.github.io/Weather-Data/Townsville.html","https://karki23.github.io/Weather-Data/Adelaide.html","https://karki23.github.io/Weather-Data/MountGambier.html","https://karki23.github.io/Weather-Data/Nuriootpa.html","https://karki23.github.io/Weather-Data/Woomera.html","https://karki23.github.io/Weather-Data/Albany.html","https://karki23.github.io/Weather-Data/Witchcliffe.html","https://karki23.github.io/Weather-Data/PearceRAAF.html","https://karki23.github.io/Weather-Data/PerthAirport.html","https://karki23.github.io/Weather-Data/Perth.html","https://karki23.github.io/Weather-Data/SalmonGums.html","https://karki23.github.io/Weather-Data/Walpole.html","https://karki23.github.io/Weather-Data/Hobart.html","https://karki23.github.io/Weather-Data/Launceston.html","https://karki23.github.io/Weather-Data/AliceSprings.html","https://karki23.github.io/Weather-Data/Darwin.html","https://karki23.github.io/Weather-Data/Katherine.html","https://karki23.github.io/Weather-Data/Uluru.html"]       

i=0
while i<49:
    scrape(lists[i])
    i=i+1      
    #print(i)   #this statement lets us know the progress of our code

import csv
with open("data.csv", "w", newline="") as f:
    writer=csv.writer(f)
    writer.writerows(my_list)

import pandas as pd
df = pd.read_csv('data.csv')
df.to_csv('dataset.csv', index=False)



