from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

#선형회귀 페이지
@app.route('/score/page1')
def score_page1():
    return render_template('page1.html')

#선형회귀 모델링
def linear():
    df=pd.read_csv('./data/LinearRegressionData.csv')  
    X=df.iloc[:, :-1].values
    y=df.iloc[:, -1].values
    reg=LinearRegression()
    reg.fit(X,y)
    return reg, X,y

#선형회귀 예측
@app.route('/score/linear')
def score_linear():
    reg = linear()[0]
    hour=int (request.args['hour'])
    pred= reg.predict([[hour]])
    return str(pred[0])


#선형회귀 그래프 
@app.route('/score/linear/graph')
def score_linear_graph():
    X = linear()[1]
    y = linear()[2]
    reg = linear()[0]
    
    plt.figure(figsize=(10,5), dpi=50)
    plt.scatter(X, y, color="blue")
    plt.plot(X, reg.predict(X), color="green")
    plt.title('Score by hours')
    plt.xlabel('hours')
    plt.ylabel('score')
    img= BytesIO()
    plt.savefig(img, format='png', dpi=50)
    img.seek(0)
    return send_file(img, mimetype='image/png')

#다중선형회귀 페이지
@app.route('/score/page2')
def score_page2():
    return render_template('page2.html')
    
    
#다중 선형회귀 모델링
@app.route('/score/multi')
def score_multi():
    place=int(request.args['place'])
    place1=0
    place2=0
    if place==1:
        place1=0
        place2=1
    elif place==2:
        place1=1
        place2=0
    hour=int(request.args['hour'])
    absent = int(request.args['absent'])
        
    #데이터 읽기
    df = pd.read_csv('./data/MultipleLinearRegressionData.csv')
    #독립/종속 변수 분리 
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    #X장소 인코딩
    ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'),[2])], remainder='passthrough')
    X = ct.fit_transform(X)
    #훈련, 테스트 세트로 분리
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    #점수 예측
    pred=reg.predict([[place1, place2, hour, absent]])
    return str(pred[0])

#다항회귀 페이지
@app.route('/score/page3')
def score_page3():
    return render_template('page3.html')

#다항회귀 모델링
def score_poly(degree):
    pass
    #데이터 읽기
    df=pd.read_csv('./data/PolynomialRegressionData.csv')
    X=df.iloc[:, :-1].values
    y=df.iloc[:, -1].values
    
    # X독립변수를 다항으로 변경
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X)
    
    #선형 객체 만들어서 학습하기
    reg = LinearRegression()
    reg.fit(X_poly, y)
    return reg, poly_reg, X, y
    
@app.route('/score/poly/graph')
def score_poly_graph():
    degree = int(request.args['degree'])
    #X범위를 0.1단위로 생성
    X = score_poly(degree)[2]
    y = score_poly(degree)[3]
    reg = score_poly(degree)[0]
    poly_reg = score_poly(degree)[1]
    
    X_range = np.arange(min(X), max(X), 0.1)
    X_range = X_range.reshape(-1, 1)
    
    #그래프 출력
    plt.figure(figsize=(10,5), dpi=90)
    plt.scatter(X, y, color="blue")
    plt.plot(X_range, reg.predict(poly_reg.fit_transform(X_range)), color="green")
    plt.title('Score by hours (genius)')
    plt.xlabel('hours')
    plt.ylabel('score')
    img= BytesIO()
    plt.savefig(img, format='png', dpi=50)
    img.seek(0)
    return send_file(img, mimetype='image/png')
    
if __name__ == '__main__':
    app.run(port=5000, debug=True)