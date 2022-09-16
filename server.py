from flask import Flask,request,jsonify

app=Flask()

model_file='./models/pipeline.bin'

with open(model_file,'rb') as f:
    pipeline=pickle.load(f)

@app.route('/predict',methods=['POST'])
def predict():
    customer=request.get_json()

    y_pred = pipeline.predict_proba(customer)[0,1]
    exited=y_pred>0.6

    result_dict={
        'exited_prob':float(y_pred),
        'exited':bool(exited)
    }

    return jsonify(result_dict)


if __name__=='main':
    app.run(host='0.0.0.0')