from flask import Flask, render_template, request
import mariadb
import os
import base64

app = Flask(__name__)

app.static_folder = 'static'

conn = mariadb.connect(
        user="testuser",
        password="3012",
        host="192.168.0.100",
        port=3306,
        database="Rats")

@app.route('/')

def index():
    conn = mariadb.connect(
        user="testuser",
        password="3012",
        host="192.168.0.100",
        port=3306,
        database="Rats")
    cur = conn.cursor()
    cur.execute("SELECT start,end,img FROM tbl_logs ORDER BY row DESC")
    results = cur.fetchall()
    for row in range (0,len(results)-1): 
        temp = (results[row][0],results[row][1], base64.b64encode(results[row][2]).decode('utf-8'))
        results[row] = temp
    
    print(type(results))
    return render_template('index.html',data=results)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)