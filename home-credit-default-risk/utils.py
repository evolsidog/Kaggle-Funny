from sklearn.metrics import roc_curve, auc, classification_report


def split_num_str_data(df):
    df_num = df.select_dtypes(include=['int64', 'float64'])
    df_str = df.select_dtypes(include='object')
    return df_num, df_str


def drop_nan_by_thresh(df, thresh):
    df = df.dropna(axis=1, thresh=int(thresh * df.shape[0]))
    return df


def estimate_auc(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred, pos_label=1)
    print("AUC ", auc(x=fpr, y=tpr))


def report_classification(y_test, predicted):
    report = classification_report(y_test, predicted)
    print("Classification report: ", report)
