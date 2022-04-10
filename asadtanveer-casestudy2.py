#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np


# In[28]:


data=pd.read_csv('casestudy.csv')


# In[29]:


data.head()


# # Total revenue for the current year

# In[30]:


data.groupby('year')['net_revenue'].sum()


# # New Customers revenue e.g. new customers not present in previous year only

# In[31]:


#revenue of new customers in 2016
customers2016 = set(data[data['year']==2016]['customer_email'])
customers2015 = set(data[data['year']==2015]['customer_email'])
newcustomers2016 = list(customers2016 - ((customers2015).intersection(customers2016)))
newcustomer2016df = data[data['customer_email'].isin(newcustomers2016)]
newcustomer2016revenue = newcustomer2016df[newcustomer2016df["year"]==2016]["net_revenue"].sum()

print(newcustomer2016revenue)


# In[32]:


#revenue of new customers in 2017
customers2017 = set(data[data['year']==2017]['customer_email'])
customers2016 = set(data[data['year']==2016]['customer_email'])
newcustomers2017 = list(customers2017 - ((customers2016).intersection(customers2017)))
newcustomer2017df = data[data['customer_email'].isin(newcustomers2017)]
newcustomer2017revenue = newcustomer2017df[newcustomer2017df["year"]==2017]["net_revenue"].sum()

print(newcustomer2017revenue)


# # Existing Customer Growth. To calculate this, use the Revenue of existing customers for current year –(minus) Revenue of existing customers from the previous year

# In[33]:


# assuming existing customers means the customer who still exists from the last year

# existing customer growth for 2017
customers2017 = set(data[data['year']==2017]['customer_email'])
customers2016 = set(data[data['year']==2016]['customer_email'])

existingcustomers2017 = list(customers2017.intersection(customers2016))
existingcustomers2017df = data[data['customer_email'].isin(existingcustomers2017)]
existingcustomergrowth2017 = existingcustomers2017df[existingcustomers2017df["year"]==2017]["net_revenue"].sum() - existingcustomers2017df[existingcustomers2017df["year"]==2016]["net_revenue"].sum()

print(existingcustomergrowth2017)


# In[34]:


# assuming existing customers means the customer who still exists from the last year

# existing customer growth for 2016
customers2016 = set(data[data['year']==2016]['customer_email'])
customers2015 = set(data[data['year']==2015]['customer_email'])

existingcustomers2016 = list(customers2016.intersection(customers2015))
existingcustomers2016df = data[data['customer_email'].isin(existingcustomers2016)]
existingcustomergrowth2016 = existingcustomers2016df[existingcustomers2016df["year"]==2016]["net_revenue"].sum() - existingcustomers2016df[existingcustomers2016df["year"]==2015]["net_revenue"].sum()

print(existingcustomergrowth2016)


# # Revenue lost from attrition

# In[35]:


# revenue lost from attrition is the total revenue that we lost from the customers we lost in subsequent years

#revenue lost from attrition in 2017
customers2017 = set(data[data['year']==2017]['customer_email'])
customers2016 = set(data[data['year']==2016]['customer_email'])
lostcustomers2017 = list(customers2016 - ((customers2017).intersection(customers2016)))
lostcustomers2017df = data[data['customer_email'].isin(lostcustomers2017)]
revenuelostfromattrition2017 = lostcustomers2017df[lostcustomers2017df["year"]==2016]["net_revenue"].sum()
print(revenuelostfromattrition2017)


# In[36]:


#revenue lost from attrition in 2016
customers2016 = set(data[data['year']==2016]['customer_email'])
customers2015 = set(data[data['year']==2015]['customer_email'])
lostcustomers2016 = list(customers2015 - ((customers2016).intersection(customers2015)))
lostcustomers2016df = data[data['customer_email'].isin(lostcustomers2016)]
revenuelostfromattrition2016 = lostcustomers2016df[lostcustomers2016df["year"]==2015]["net_revenue"].sum()
print(revenuelostfromattrition2016)


# # Existing Customers revenue current year

# In[37]:


# assuming existing customers means the customer who still exists from the last year
# existing customer revenue for 2017
customers2017 = set(data[data['year']==2017]['customer_email'])
customers2016 = set(data[data['year']==2016]['customer_email'])

existingcustomers2017 = list(customers2017.intersection(customers2016))
existingcustomers2017df = data[data['customer_email'].isin(existingcustomers2017)]
existingcustomerrevenuecuryear2017 = existingcustomers2017df[existingcustomers2017df["year"]==2017]["net_revenue"].sum() 
print(existingcustomerrevenuecuryear2017)


# In[38]:


# assuming existing customers means the customer who still exists from the last year
# existing customer growth for 2016
customers2016 = set(data[data['year']==2016]['customer_email'])
customers2015 = set(data[data['year']==2015]['customer_email'])

existingcustomers2016 = list(customers2016.intersection(customers2015))
existingcustomers2016df = data[data['customer_email'].isin(existingcustomers2016)]
existingcustomerrevenuecuryear2016 = existingcustomers2016df[existingcustomers2016df["year"]==2016]["net_revenue"].sum()
print(existingcustomerrevenuecuryear2016)


# # Existing Customers revenue previous year

# In[39]:


# assuming existing customers means the customer who still exists from the last year
# existing customer revenue for 2017
customers2017 = set(data[data['year']==2017]['customer_email'])
customers2016 = set(data[data['year']==2016]['customer_email'])

existingcustomers2017 = list(customers2017.intersection(customers2016))
existingcustomers2017df = data[data['customer_email'].isin(existingcustomers2017)]
existingcustomerrevenueprevyear2017 = existingcustomers2017df[existingcustomers2017df["year"]==2016]["net_revenue"].sum() 
print(existingcustomerrevenueprevyear2017)


# In[40]:


# assuming existing customers means the customer who still exists from the last year
# existing customer growth for 2016
customers2016 = set(data[data['year']==2016]['customer_email'])
customers2015 = set(data[data['year']==2015]['customer_email'])

existingcustomers2016 = list(customers2016.intersection(customers2015))
existingcustomers2016df = data[data['customer_email'].isin(existingcustomers2016)]
existingcustomerrevenueprevyear2016 = existingcustomers2016df[existingcustomers2016df["year"]==2015]["net_revenue"].sum()
print(existingcustomerrevenueprevyear2016)


# # Total Customers current year

# In[41]:


# total customers current year for 2017
customers2015 = data[data['year']==2015]
len(customers2015)


# In[42]:


# total customers current year for 2016
customers2016 = data[data['year']==2016]
len(customers2016)


# In[43]:


# total customers current year for 2017
customers2017 = data[data['year']==2017]
len(customers2017)


# # Total Customers Previous Year

# In[44]:


#total customer previous year for 2017
customers2017prevyear = data[data['year']==2016]
len(customers2017prevyear)


# In[45]:


#total customers previous year for 2016
customers2016prevyear = data[data['year']==2015]
len(customers2016prevyear)


# # New Customers

# In[46]:


#new customers in 2016
customers2016 = set(data[data['year']==2016]['customer_email'])
customers2015 = set(data[data['year']==2015]['customer_email'])
newcustomers2016 = (customers2016 - ((customers2015).intersection(customers2016)))
newcustomers2016


# In[47]:


#new customers in 2017
customers2017 = set(data[data['year']==2017]['customer_email'])
customers2016 = set(data[data['year']==2016]['customer_email'])
newcustomers2017 = customers2017 - ((customers2017).intersection(customers2016))
newcustomers2017


# # Lost Customers

# In[48]:


#lost customers in 2016
customers2016 = set(data[data['year']==2016]['customer_email'])
customers2015 = set(data[data['year']==2015]['customer_email'])
lostcustomers2016 = (customers2015 - ((customers2015).intersection(customers2016)))
lostcustomers2016


# In[49]:


#lost customers in 2017
customers2017 = set(data[data['year']==2017]['customer_email'])
customers2016 = set(data[data['year']==2016]['customer_email'])
lostcustomers2017 = (customers2016 - ((customers2017).intersection(customers2016)))
lostcustomers2017


# 

# # Plots from the information

# In[50]:


# plotting lineplots between year and total_customers
import matplotlib.pyplot as plt
years = ["2015","2016","2017"]
total_customers = [len(customers2015),len(customers2016),len(customers2017)]
plt.plot(years,list(total_customers), color='r', label='total_customers')
plt.xlabel('year')
plt.ylabel('total_customers')
plt.legend()
plt.show()


# As we can see from the line plot the number of customers dropped from 2015 to 2016 but increased again in 2017.

# In[51]:


# plotting lineplots between year and total_customers
import matplotlib.pyplot as plt
years = ["2015","2016","2017"]
total_revenue = [29036749.19,25730943.59,31417495.03]
plt.plot(years,total_revenue, color='g', label='total_revenue')
plt.xlabel('year')
plt.ylabel('total_revenue')
plt.legend()
plt.show()


# As we can see the shape of the graph is similar to that of total customers in that year.Hence we assume that the total_cutomers is highly correlated to revenue that year.

# In[52]:


import seaborn as sns
data_viz = {'index': [0, 1],
        'new_customers': [len(newcustomers2016),len(newcustomers2017)],
        'lost_customers': [len(lostcustomers2016),len(lostcustomers2017)]}

df = pd.DataFrame(data_viz)

year = ["2016","2017"]

plt.figure(figsize=(16,10));
sns.set_style('ticks')
ax = sns.barplot(data=df,                  x='index',                  y='new_customers',                  hue='lost_customers',                  palette=sns.color_palette("Reds_d", n_colors=7, desat=1))

ax.set_xlabel("Year", fontsize=18, alpha=0.8)
ax.set_ylabel("new customers", fontsize=18, alpha=0.8)
ax.set_title("new customers each year", fontsize=24)
ax.set_xticklabels(year, fontsize=16)
ax.legend(fontsize=15)
sns.despine()
plt.show()


# The bar plot helps us visualize and compare new customers that were gained in 2016 and 2017.The color hue is proportional to the customers lost that year.This visualization easily helps us see that:
# 
#     1- new customers in 2017 are higher in number in 2016.
#     
#     2- The color hue helps us understand that new customers is positively correlated to lost customers because as the number of new customers increase the number of lost customers increase as well.

# In[ ]:




