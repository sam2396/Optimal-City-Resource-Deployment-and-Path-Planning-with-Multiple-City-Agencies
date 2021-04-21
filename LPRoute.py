import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pulp import *
import haversine as hs


df=pd.read_csv('../../../Datasets/cop_data.csv')
df['coordinate'] = list(zip(df['Latitude'],df['Longitude']))


def dist_from(loc1,loc2):
    dist=hs.haversine(loc1,loc2)
    return round(dist,2)

# parse location one by one to dist_from
for _,row in df.iterrows():
    df[row.ID]=df['coordinate'].apply(lambda x: dist_from(row.coordinate,x))

distances_df=df.iloc[:,19:4183]
#distances_df.index=df.ID
distances_df.insert(0, 'ID', df.ID)


#positions to be plotted
locations = dict( ( ID, (df[['ID','Latitude','Longitude']].loc[ID, 'Longitude'], df[['ID','Latitude','Longitude']].loc[ID, 'Latitude']) ) for ID in df.index)

for l in locations:
    lo = locations[l]
    plt.plot(lo[0], lo[1], 'o')
    plt.text(lo[0] + .01, lo[1], l, horizontalalignment='center', verticalalignment='center')

plt.gca().axis('off');

# get distance in a dictionary form
distances = dict( ((l1,l2), distances_df.iloc[l1, l2] ) for l1 in locations for l2 in locations if l1!=l2)

V = 3 #the number vehicles/people deployed

#problem
prob=LpProblem("vehicle", LpMinimize)
#indicates if location i is connected to location j along route
indicator = LpVariable.dicts('indicator',distances, 0,1,LpBinary)
#elimiate subtours
eliminator = LpVariable.dicts('eliminator', df.ID, 0, len(df.ID)-1, LpInteger)

# constraints
for v in df.ID:
    cap = 1 if v != '9aed9ae5ee7f83638ee33ccff49cf6b9' else V
    # inward possible route
    prob += lpSum([indicator[(i, v)] for i in df.ID if (i, v) in indicator]) == cap
    # outward possible route
    prob += lpSum([indicator[(v, i)] for i in df.ID if (v, i) in indicator]) == cap

# subtour elimination
num = len(df.ID) / V
for i in df.ID:
    for j in df.ID:
        if i != j and (i != '9aed9ae5ee7f83638ee33ccff49cf6b9' and j != '9aed9ae5ee7f83638ee33ccff49cf6b9') and (
        i, j) in indicator:
            prob += eliminator[i] - eliminator[j] <= (num) * (1 - indicator[(i, j)]) - 1



%time prob.solve()
print(LpStatus[prob.status])

feasible_edges = [ e for e in indicator if value(indicator[e]) != 0 ]

def get_next_loc(initial):
    '''to get the next edge'''
    edges = [e for e in feasible_edges if e[0]==initial]
    for e in edges:
        feasible_edges.remove(e)
    return edges

routes = get_next_loc('9aed9ae5ee7f83638ee33ccff49cf6b9')
routes = [ [e] for e in routes ]

for r in routes:
    while r[-1][1] !='9aed9ae5ee7f83638ee33ccff49cf6b9':
        r.append(get_next_loc(r[-1][1])[-1])

for r in routes:
    print(' -> '.join([ a for a,b in r]+['9aed9ae5ee7f83638ee33ccff49cf6b9']))

#outline the routes
coloured_loc = [np.random.rand(3) for i in range(len(routes))]
for r,co in zip(routes,coloured_loc):
    for a,b in r:
        l1,l2 = locations[a], locations[b]
        plt.plot([l1[0],l2[0]],[l1[1],l2[1]], coloured_loc=co)

for l in locations:
    lo = locations[l]
    plt.plot(lo[0], lo[1], 'o')
    plt.text(lo[0] + .01, lo[1], l, horizontalalignment='center', verticalalignment='center')

plt.title('%d ' % V + 'Vehicle routes' if V > 1 else 'Vehicle route')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()