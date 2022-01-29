# Import required libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import base64

# Configures the default setting of the page
st.set_page_config(
     page_title="ML APP Supervised",
     page_icon=":desktop_computer",
     layout="wide",
     initial_sidebar_state="expanded"
 )

 # Page Title
st.title('Machine Learning Application - Supervised')
st.markdown('<b>This application takes a csv or excel file, do data exploration, data pre-processing and create supervised machine learning model!</b>', unsafe_allow_html=True)

# Upload a file and read the data
data_file = st.file_uploader('Select a file', ['csv', 'xlsx', 'xlsb', 'xlsm'], help='Only csv or excel files')

# Create an empty dataframe
df = pd.DataFrame()
if data_file is not None:
    if data_file.name.endswith('csv'):
        df = pd.read_csv(data_file)
    else:
        df = pd.read_excel(data_file)

# Create two columns/containers two show meta data information and data summary
if df.empty == False:
    st.header('Data Exploration')
    # Show Dataframe button
    #display_data = st.checkbox('Display Raw Data')
    with st.expander('Show/Hide Raw Data'):
        st.dataframe(df.astype('str'))
    col1, col2 = st.columns([1, 2])
    col1.write('#### Meta Data')
    col1.dataframe((pd.DataFrame([df.dtypes, df.count()], index=['Dtype', 'Non-Null Count']).T).astype('str'))
    col2.write('#### Data Summary')
    col2.dataframe(df.describe().astype('str'))

    # Modeling
    st.header('Modeling')
    col1, col2, col3 = st.columns(3)

    col1.write('#### Supervised Learning')
    # Select Regression or Classification
    reg_class = col1.selectbox('Type', ['Regression', 'Classification'])
    col1.write('Below algorithms will be used in modeling')
    # Regression and Classification list
    if reg_class=='Regression':
        regressionAlgo = ['Linear Regression', 'SVR', 'KNN', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
        col1.write('  \n'.join(regressionAlgo))
    else:
        regressionAlgo = ['Logistic Regression', 'SVC', 'KNN', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
        #col1.write('Logistic Regression, SVC, KNN, Decision Tree, Random Forest, Gradient Boosting')
        col1.write('  \n'.join(regressionAlgo))

    col2.write('#### Parameters')
    # Parameters for different algorithms
    n_neighbor = col2.number_input('Number of nearest neighbor for KNN', min_value=1, value=5, max_value=50, help='Range 1 to 50')
    n_trees = col2.number_input('Number of trees for Random Forest', min_value=1, value=10, max_value=50, help='Range 1 to 50')
    n_estimator = col2.number_input('Number of estimators for Gradient Boosting', min_value=1, value=10, max_value=50, help='Range 1 to 50')

    col3.write('#### Dependent/Independent Variables')
    dfColumns = df.columns
    columnsType_d = dict(df.dtypes)
    response = col3.selectbox('Response/Dependent Variable', dfColumns)
    predictor = col3.multiselect('Predictors/Independent Variables', [i for i in dfColumns if i != response], 
    [i for i in dfColumns if i != response])

    st.write('#### Data Visualization')
    #st.markdown("<h5 style='text-align: center;'>Data Visualization</h5>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # Finding Correlation between Numerical columns
    xCorr = df[predictor].corr()
    # Plotting a correlation heatmap
    fig1 = px.imshow(xCorr, color_continuous_scale='RdBu')
    fig1.update_layout(
        plot_bgcolor='white', 
        xaxis={'showgrid': False}, 
        yaxis={'showgrid': False}, 
        title_text='Correlation Plot (Only Continuous Variables)', title_x=0.5, 
        title_font_color="gray",
        title_font_size=24
    )
    col1.plotly_chart(fig1, use_container_width=True)

    # label information
    if reg_class == 'Regression':
        fig2 = px.histogram(df, x=response, title='Distribution by ' + response + ' (Must be Continuous variable)')
    else:
        df_count = df[response].value_counts().reset_index()
        df_count.columns = [response, 'Count']
        fig2 = px.bar(df_count, x=response, y='Count', title='Distribution by ' + response + ' (Must be Categorical Variable)')
    fig2.update_layout(
        plot_bgcolor='white', 
        #xaxis={'showgrid': False}, 
        #yaxis={'showgrid': False},
        title_x=0.5, 
        title_font_color="gray",
        title_font_size=24
    )
    col2.plotly_chart(fig2, use_container_width=True)

    st.write('#### Data Pre-Processing')
    col1, col2, col3, col4 = st.columns(4)
    scaling = col1.radio('Feature Scaling Technique', ('Standardization', 'Normalization'), 0)
    impute_num = col2.radio('Missing - Impute Method for Numerical', ('mean', 'median'), 0)
    impute_cat = col3.radio('Missing - Impute Method for Categorical', ('most_frequent', 'constant - not available'), 0)

    # Button to start model building process
    if st.button('Build Models'): 
        try:
            # Import train test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(df[predictor], df[response], test_size=0.25, random_state=100)

            # Import required libraries for data pre-processing
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer

            scaling_d = {'Standardization': StandardScaler(), 'Normalization': MinMaxScaler()}
            if scaling == 'Standardization':
                scaler = StandardScaler()
            else:
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()

            # Finding numeric and categorical features
            numericFeatures = X_train.select_dtypes(include=['int64', 'float64', 'datetime64', 'timedelta64']).columns
            categoricalFeatures = X_train.select_dtypes(include=['object', 'bool', 'category']).columns

            # Create Transfomer for numerical and categorical variables
            numeric_tranformer = Pipeline(
                steps=[('imputer', SimpleImputer(strategy=impute_num)), 
                ('scaler', scaler)]
            )

            if impute_cat == 'constant - not available':
                categorical_transformer = Pipeline(
                    steps=[('imputer', SimpleImputer(strategy='constant', fill_value='not available')), 
                    ('onehotencoding', OneHotEncoder(drop='first'))]
            )
            else:
                categorical_transformer = Pipeline(
                    steps=[('imputer', SimpleImputer(strategy=impute_cat)), 
                    ('onehotencoding', OneHotEncoder(drop='first'))]
                )

            pre_processor = ColumnTransformer(
                transformers=[
                    ('numerical', numeric_tranformer, numericFeatures),
                    ('categorical', categorical_transformer, categoricalFeatures)
                ]
            )

            # Build Models
            # Import all required libraries for machine learning models
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.svm import SVR, SVC
            from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
            from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
            from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_recall_fscore_support

            allRegressor_d = {
                'Linear Regression': LinearRegression(),
                'SVR': SVR(), 
                'KNN': KNeighborsRegressor(n_neighbors=n_neighbor),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(n_estimators=n_trees),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=n_estimator)
            }
            allClassifer_d = {
                'Logistic Regression': LogisticRegression(),
                'SVC': SVC(),
                'KNN': KNeighborsClassifier(n_neighbors=n_neighbor),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(n_estimators=n_trees),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=n_estimator)
            }

            def model_building(xdict, pp = pre_processor, X=X_train, y=y_train):
                model_dict = dict()
                for k,v in xdict.items():
                    pipe = Pipeline(
                        steps=[('preprocessing', pp), 
                        ('model', v)]
                    )
                    pipe.fit(X, y)

                    # Add pipe to model_dict
                    model_dict[k] = pipe

                # Return all the model in dictionary with their name
                return model_dict

            if reg_class=='Regression':
                models_d = allRegressor_d
            else:
                models_d = allClassifer_d
            
            # Call the function
            all_models = model_building(models_d)

            # Function to calculate the metrics
            def metrics_out(actual, pred, rc=reg_class):
                if rc == 'Regression':
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    metrics_calc = {'MAE': round(mean_absolute_error(actual, pred), 2), 
                    'MSE': round(mean_squared_error(actual, pred, squared=True), 2), 
                    'RMSE': round(mean_squared_error(actual, pred, squared=False), 2)}
                else:
                    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
                    prfs = precision_recall_fscore_support(actual, pred, average='macro')
                    metrics_calc = {
                        'Accuracy': round(accuracy_score(actual, pred), 2),
                        'Precision': round(prfs[0], 2),
                        'Recall': round(prfs[1], 2), 
                        'F-Score': round(prfs[2], 2)
                    }

                return metrics_calc

            # Function to check model performance
            def model_performance(mod=all_models, X=X_test, y=y_test):
                performance_d = dict()
                for k,v in mod.items():
                    y_pred = v.predict(X)
                    performance_d[k] = metrics_out(y, y_pred)
                return pd.DataFrame(performance_d)

            performance_df = model_performance()

            st.write('#### Ouput')

            col1, col2, col3, col4 = st.columns([3, 1, 2, 1])
            col1.write('Done Building Models  \nThe performance for each model is provided in the table')

            col1.dataframe(performance_df.astype('str').T)

            def download_model(k, v, col):
                output_model = pickle.dumps(v)
                b64 = base64.b64encode(output_model).decode()
                href = f'<a href="data:file/output_model;base64,{b64}" download="{k}.pkl">{k} as pkl File</a>'
                col.markdown(href, unsafe_allow_html=True)

            col3.write('Download Trained Model')
            for k,v in all_models.items():
                download_model(k, v, col3)
        except:
            st.error('Some Error in data and/or selection. Please re-check data and make the correct selection!')


