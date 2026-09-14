




def impute_data(train_data, test_data):

    train_data['Tratamiento_num'] = train_data['Tratamiento'].str.extract(r'T(\d+)').astype(float)
    test_data['Tratamiento_num'] = test_data['Tratamiento'].str.extract(r'T(\d+)').astype(float)

    min_frac_no_nulos = 0.05
    df_train_numeric = train_data.select_dtypes(include=[float, int])
    df_train_numeric = df_train_numeric.loc[:, df_train_numeric.isna().mean() < (1 - min_frac_no_nulos)]

    df_test_numeric = test_data.select_dtypes(include=[float, int])
    df_test_numeric = df_test_numeric.loc[:, df_test_numeric.isna().mean() < (1 - min_frac_no_nulos)]

    limites_clip = df_train_numeric.quantile([0.01, 0.99]).T
    limites_clip.columns = ['min_val', 'max_val']

    df_train_numeric_original = df_train_numeric.copy()


    for tratamiento in df_train_numeric['Tratamiento_num'].unique():
        mask_train = df_train_numeric['Tratamiento_num'] == tratamiento
        mask_test = df_test_numeric['Tratamiento_num'] == tratamiento

        grupo_train = df_train_numeric.loc[mask_train].drop(columns=['Tratamiento_num'])
        grupo_test = df_test_numeric.loc[mask_test].drop(columns=['Tratamiento_num'])



    def imputar_grupo(group):
        columnas_validas = group.columns[~group.isna().all()]
        group_valido = group[columnas_validas]

        # GUARDAR MÁSCARA DE VALORES ORIGINALES
        mascara_original = ~group_valido.isna()
        valores_originales = group_valido.copy()
        
        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=42), 
                                max_iter=20, random_state=42)
        imputado = imputer.fit_transform(group_valido)
        
        df_imputado = pd.DataFrame(imputado, columns=columnas_validas, index=group.index)
        
        # RESTAURAR VALORES ORIGINALES (solo imputar los NaN)
        df_imputado[mascara_original] = valores_originales[mascara_original]
        
        df_resultado = pd.DataFrame(index=group.index, columns=group.columns)
        df_resultado[columnas_validas] = df_imputado
        return df_resultado




