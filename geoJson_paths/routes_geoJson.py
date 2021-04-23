import folium
from folium import IFrame
import os
import base64

m = folium.Map(location=[43.084087245743184, -77.68043160438538], zoom_start=17)
routeData = os.path.join('trial_Loop.json')

folium.GeoJson(routeData, name='route1').add_to(m)



# Template to add a marker and image to the nodes
tooltip = "Click to see picture"
html = '<img src="data:image/png;base64,{}">'.format

picture1 = base64.b64encode(open('./images/el_tigre.png','rb').read()).decode()
iframe1 = IFrame(html(picture1), width=600+20, height=400+20)
popup1 = folium.Popup(iframe1, max_width=650)
icon1 = folium.Icon(color="orange", icon="glyphicon-home")
marker1 = folium.Marker(location=[43.084087245743184, -77.68043160438538], popup=popup1, tooltip=tooltip, icon=icon1).add_to(m)

m.save("indexFinal.html")