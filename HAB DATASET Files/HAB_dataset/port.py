import mysql.connector

mydb  = mysql.connector.connect(
    host="10.32.84.33",
    user = "vifapi",
    port="3306",
    password = "braunschweig"
)

cursor = mydb.cursor()
cursor.execute("CREATE DATABASE hab_dataset")