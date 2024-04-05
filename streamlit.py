import streamlit as st 
import pandas as pd
import pickle
import io
import pandas as pd
from utility import *


@st.cache_data 
def load_dataRFM():
    data = pd.read_csv('data/OnlineRetail_RFM.csv', index_col=0)
    return data
@st.cache_data
def load_data():
    data = pd.read_csv('data/OnlineRetailCleaned.csv', index_col=0)
    return data
@st.cache_resource
def load_model(): 
    with open('models/kmeansLDS6.pkl', 'rb') as f:
        model = pickle.load(f) 
    return model
@st.cache_resource
def load_scaler(): 
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f) 
    return scaler



def add_data(R, F, M):
    new_index = len(st.session_state['data'])
    st.session_state['data'].loc[new_index] = [R, F, M]


dataRFM = load_dataRFM()
dataOrders = load_data()
scaler = load_scaler()
model = load_model()
st.image('images/topic.png', caption='Shoppe')
st.write("""# Customer Segmentation for Online Retail with RFM""")  

menu = ["Overview", "About Project" , "Tìm kiếm khách hàng", "New Predict by RFM"]
choice = st.sidebar.selectbox('Danh mục', menu)
if choice == 'Overview': 
    st.write("""### Thành viên nhóm:
- Huỳnh Văn Tài  
- Trần Thế Lâm""") 
    st.write("""### Mô tả dự án:
    - Sử dụng dữ liệu từ doanh số bán hàng của công ty, 
    - Tiến hành xử lý và phân cụm khách hàng dự trên các thuật toán phân cụm: Quantiles,  K-mean scikit learn - Hierarchical - K-mean Spark 
    - Phân cụm khách hàng để phù hợp với chiến lược của công ty
            """)
    st.write("""### Mục tiêu:  
    - Phân cụm khách hàng dựa trên RFM  
    - Xem thông tin phân khúc của khách hàng đã có sẵn  
    - Dự đoán phân khúc cho khách hàng mới dựa vào RFM""")

    st.write("""### Mô tả dữ liệu:
    - Dữ liệu gốc: OnlineRetail.csv
    - Dữ liệu đã xử lý: OnlineRetailCleaned.csv
    - Dữ liệu RFM: OnlineRetail_RFM.csv
        """)

elif choice == 'About Project':
    st.header('Dự án Phân tích Dữ liệu Bán hàng')

    st.subheader('Dữ liệu bán hàng Cleaned (Sample)')
    st.dataframe(dataOrders.head(10))
    st.subheader('Dữ liệu đã xử lý RFM')
    st.dataframe(dataRFM.head(10))
    st.subheader('Thống kê dữ liệu')
    st.text(f'Số lượng dòng và cột: {dataOrders.shape}')
    st.dataframe(dataOrders.describe())

    st.subheader('EDA cơ bản')
    st.subheader('Top 10 sản phẩm bán chạy nhất')
    st.image('images/top_10_san_pham_ban_chay_nhat.png')

    st.subheader('Thống kê doanh số bán hàng theo tháng')
    st.image('images/doanh_thu_theo_thang.png')
    st.markdown('Biểu đồ doanh số theo tháng giúp nhận diện xu hướng và mẫu thời gian trong doanh số bán hàng.')

    st.subheader('Thống kê doanh số bán hàng theo quốc gia')
    st.image('images/revenue_by_country.png', caption='Doanh số bán hàng theo quốc gia')
    st.markdown('Biểu đồ này phân tích doanh số bán hàng dựa trên địa lý quốc gia.')

    st.subheader('Phân phối RFM Kmean')
    st.image('images/Custom RFM Segments Kmean LDS6 - Tree Map.png')
    st.markdown('### Chú thích:')
    st.markdown("""
    - Cluster 0: Lost - Khách hàng chỉ mua 1 lần
    - Cluster 1: Big spender - Nhóm khách hàng mua hàng nhiều và thường xuyên, với số tiền lớn
    - Cluster 2: At risk - Nhóm khách hàng có nguy cơ rời bỏ - cần được chăm sóc
    - Cluster 3: Regular - Nhóm khách hàng đã từng mua và quay lại mua nhiều hơn 2 lần
    """)




elif choice == 'Tìm kiếm khách hàng':
    options = st.radio('Chọn phương pháp', ["Nhập thủ công", "Input từ file txt"])
    if options == "Nhập thủ công":
        customer_id_options = dataRFM.index.astype(int).tolist()
        customer_id_select = st.multiselect('CustomerID', customer_id_options)

        submit_button = st.button('Xem thông tin')
        if submit_button:
            if customer_id_select:
                customer_id_select = [int(i) for i in customer_id_select]
                df_temp = dataRFM[dataRFM.index.isin(customer_id_select)]
                df_result = customer_segmentKmean(df_temp, scaler, model)
                st.write(df_result)
                st.download_button(label='Download', data=df_result.to_csv(), file_name='customerSegnmentationwRFM.csv', mime='text/csv')
            else: 
                st.write('Chưa chọn CustomerID')
    else:
        st.write("""### Tìm kiếm khách hàng trong dataset""")
        st.markdown("""
            - Upload file txt chứa danh sách CustomerID  
            - Mỗi dòng là 1 CustomerID  
            """)
        uploaded_file = st.file_uploader("Upload Files", type=['txt'])
        if uploaded_file is not None:
            file = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            try:
                customerIDs = file.read().split('\n')
                customerIDs = [int(i) for i in customerIDs if i]
                df_temp = dataRFM[dataRFM.index.isin(customerIDs)]
                df_result = customer_segmentKmean(df_temp, scaler, model)
                st.write(df_result)
                st.download_button(label='Download', data=df_result.to_csv(), file_name='customerSegnmentationwRFM.csv', mime='text/csv')
            except Exception as e:
                st.write("Xảy ra lỗi trong quá trình đọc file - hoặc file không đúng định dạng")
elif choice == 'New Predict by RFM':
    options = st.radio('Chọn phương pháp', ["Nhập thủ công", "Input từ file csv"])
    if options == "Nhập thủ công":
        st.write("""### Dự đoán cho khách hàng mới dựa vào RFM""")
        maxRecency = dataRFM['Recency'].max()
        maxFrequency = dataRFM['Frequency'].max()
        maxMonetary = dataRFM['Monetary'].max()
        R = st.slider('Recency', 0, maxRecency, 0)
        F = st.slider('Frequency', 0, maxFrequency, 0)
        M = st.slider('Monetary', 0, 10000, 0)
        columns = st.columns([1,1,2])
        with columns[0]: 
            submit_button = st.button('Thêm vào danh sách')
        with columns[1]:
            clear_button = st.button('Xóa danh sách')
        
        if 'data' not in st.session_state:
            st.session_state['data'] = pd.DataFrame(columns=['Recency', 'Frequency', 'Monetary'])

        
        if submit_button:
            add_data(R, F, M)

        if clear_button:
            st.session_state['data'] = pd.DataFrame(columns=['Recency', 'Frequency', 'Monetary'])

        st.write("Dữ liệu hiện tại trong danh sách:")
        st.write(st.session_state['data'])

        submit = st.button("Dự đoán")
        if submit:
            if st.session_state['data'].shape[0] > 0:
                cluster = predict_new_RFM(st.session_state['data'].copy(), scaler, model)
                st.write(cluster)
                st.download_button(label='Download', data=cluster.to_csv(), file_name='predictNewCustomerRFM.csv', mime='text/csv')
            else:
                st.write("Chưa có dữ liệu")
    else: 
        st.write("""### Dự đoán trong dataset""")
        st.markdown("""
            - Upload file .csv chứa danh sách RFM  
            - Đòng đầu tiên là header - Recency, Frequency, Monetary
            - Mỗi dòng là 1 dòng RFM  
            - Các cột cách nhau bởi dấu phẩy  
            """)
        uploaded_file = st.file_uploader("Upload Files", type=['csv'])
        if uploaded_file is not None:
            file = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            button_predict = st.button('Dự đoán')
            if button_predict:
                try:
                    df = pd.read_csv(file, index_col=0)
                    cluster = predict_new_RFM(df, scaler, model)
                    st.write(cluster)
                    st.download_button(label='Download', data=cluster.to_csv(), file_name='predictNewCustomerRFM.csv', mime='text/csv')
                except Exception as e:
                    st.write(e)


