# Ứng dụng 1. Nhập CustomerID để xác định segment của khách hàng
def customer_segmentKmean(rfm_data, scaler, model):
    rfm_scaled = scaler.transform(rfm_data)
        
    # Dùng model để dự đoán phân khúc
    cluster = model.predict(rfm_scaled)
    # Thêm cột cluster vào dataframe
    rfm_data['Segment'] = cluster
    rfm_data['Segment'] = rfm_data['Segment'].map({0:'Lost', 1:'Big spender', 2:'At risk', 3:'Regular'})    
    return rfm_data

# Input 1 mảng RFM - xuất ra 1 dataframe mới với dự đoán
def predict_new_RFM(rfm_values, scaler, model):
    rfm_values_scaler = scaler.transform(rfm_values)
    cluster = model.predict(rfm_values_scaler)
    rfm_values['Segment'] = cluster
    rfm_values['Segment'] = rfm_values['Segment'].map({0:'Lost', 1:'Big spender', 2:'At risk', 3:'Regular'})
    return rfm_values