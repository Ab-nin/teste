import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys
import os

# Função para carregar e treinar os modelos
def load_models():
    # Carregar e treinar o modelo CN
    data_cn = pd.read_csv("dados_cn.csv")
    data_cn = data_cn.dropna()
    x_cn = data_cn.drop(["CN"], axis=1)
    y_cn = data_cn["CN"]
    xtrcn, xvalcn, ytrcn, yvalcn = train_test_split(
        x_cn, y_cn, test_size=0.2, random_state=0
    )
    model_cn = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    model_cn.fit(xtrcn, ytrcn)

    # Carregar e treinar o modelo de Densidade
    data_d = pd.read_csv("dados_biodiesel_densidade.csv")
    data_d = data_d.dropna()
    x_d = data_d.drop(["Densidade"], axis=1)
    y_d = data_d["Densidade"]
    xtrd, xvald, ytrd, yvald = train_test_split(x_d, y_d, test_size=0.2, random_state=0)
    model_densidade = RandomForestRegressor(
        n_estimators=1000, random_state=0, n_jobs=-1
    )
    model_densidade.fit(xtrd, ytrd)

    # Carregar e treinar o modelo de HHV
    data_hhv = pd.read_csv("dados_biodiesel_HHV.csv")
    data_hhv = data_hhv.dropna()
    x_hhv = data_hhv.drop(["HHV"], axis=1)
    y_hhv = data_hhv["HHV"]
    xtrhhv, xvalhhv, ytrhhv, yvalhhv = train_test_split(
        x_hhv, y_hhv, test_size=0.2, random_state=0
    )
    model_hhv = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    model_hhv.fit(xtrhhv, ytrhhv)

    # Carregar e treinar o modelo de OS
    data_os = pd.read_csv("dados_biodiesel_OS.csv")
    data_os = data_os.dropna()
    x_os = data_os.drop(["OS"], axis=1)
    y_os = data_os["OS"]
    xtros, xvalos, ytros, yvalos = train_test_split(
        x_os, y_os, test_size=0.2, random_state=0
    )
    model_os = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    model_os.fit(xtros, ytros)

    # Carregar e treinar o modelo de VI
    data_iv = pd.read_csv("dados_biodiesel_IV.csv")
    data_iv = data_iv.dropna()
    x_iv = data_iv.drop(["IV"], axis=1)
    y_iv = data_iv["IV"]
    xtriv, xvaliv, ytriv, yvaliv = train_test_split(
        x_iv, y_iv, test_size=0.2, random_state=0
    )
    model_iv = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    model_iv.fit(xtriv, ytriv)

    # Carregar e treinar o modelo de FSCL
    data_fscl = pd.read_csv("dados_biodiesel_LCSF.csv")
    data_fscl = data_fscl.dropna()
    x_fscl = data_fscl.drop(["LCSF"], axis=1)
    y_fscl = data_fscl["LCSF"]
    xtrfscl, xvalfscl, ytrfscl, yvalfscl = train_test_split(
        x_fscl, y_fscl, test_size=0.2, shuffle=True, random_state=0
    )
    model_fscl = LinearRegression()
    model_fscl.fit(xtrfscl, ytrfscl)

    # Carregar e treinar o modelo de CFPP
    data_cfpp = pd.read_csv("dados_biodiesel_CFPP.csv")
    data_cfpp = data_cfpp.dropna()
    x_cfpp = data_cfpp.drop(["CFPP"], axis=1)
    y_cfpp = data_cfpp["CFPP"]
    xtrcfpp, xvalcfpp, ytrcfpp, yvalcfpp = train_test_split(
        x_cfpp, y_cfpp, test_size=0.2, shuffle=True, random_state=0
    )
    model_cfpp = LinearRegression()
    model_cfpp.fit(xtrcfpp, ytrcfpp)

    # Carregar e treinar o modelo de SV
    data_sv = pd.read_csv("dados_biodiesel_SV.csv")
    data_sv = data_sv.dropna()
    x_sv = data_sv.drop(["SV"], axis=1)
    y_sv = data_sv["SV"]
    xtrsv, xvalsv, ytrsv, yvalsv = train_test_split(
        x_sv, y_sv, test_size=0.2, shuffle=True, random_state=0
    )
    model_sv = LinearRegression()
    model_sv.fit(xtrsv, ytrsv)

    # Carregar e treinar o modelo de PCI
    data_lhv = pd.read_csv("dados_biodiesel_LHV(I).csv")
    data_lhv = data_lhv.dropna()
    x_lhv = data_lhv.drop(["LHV(I)"], axis=1)
    y_lhv = data_lhv["LHV(I)"]
    xtrlhv, xvallhv, ytrlhv, yvallhv = train_test_split(
        x_lhv, y_lhv, test_size=0.2, shuffle=True, random_state=0
    )
    model_lhv = LinearRegression()
    model_lhv.fit(xtrlhv, ytrlhv)

    # Carregar e treinar o modelo de Viscosidade
    data_v = pd.read_csv("dados_biodiesel_viscosidade.csv")
    data_v = data_v.dropna()
    x_v = data_v.drop(["Viscosidade"], axis=1)
    y_v = data_v["Viscosidade"]
    xtrv, xvalv, ytrv, yvalv = train_test_split(
        x_v, y_v, test_size=0.2, shuffle=True, random_state=0
    )
    model_viscosidade = LinearRegression()
    model_viscosidade.fit(xtrv, ytrv)

    return {
        "model_cn": model_cn,
        "model_densidade": model_densidade,
        "model_hhv": model_hhv,
        "model_os": model_os,
        "model_iv": model_iv,
        "model_fscl": model_fscl,
        "model_cfpp": model_cfpp,
        "model_sv": model_sv,
        "model_lhv": model_lhv,
        "model_viscosidade": model_viscosidade,
    }


# Carregar os modelos
models = load_models()


# Função principal do aplicativo para customização
def app():

    # Definindo a configuração da página
    icone_path = os.path.join(os.getcwd(), "logo_app.ico")
    st.set_page_config(
        page_title="Biodiesel Properties Predictor", page_icon=icone_path
    )

    # Função para carregar e aplicar o CSS
    def load_css(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Carregar o CSS
    css_path = "styles.css"
    load_css(css_path)

    # Criando o sidebar
    with st.sidebar:
        st.title("About the Software")

        st.markdown(
            "<p style='text-align:justify;'>The BPP is a software based on machine learning models designed to estimate the main properties of biodiesel exclusively produced from microalgae biomass. The program requires that users inform the value of which fatty acid solicited.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:justify;'>BPP was developed by the members of the Bioenergy and Environment Laboratory of CEFET/RJ campus Angra dos Reis: Abner Vieira Pereira, graduation student, M.Sc Paulo Victor Gomes dos Santos and Ph.D Carla Cristina Almeida Loures.</p>",
            unsafe_allow_html=True,
        )

    # Criando um container
    with st.container():
        # Criando duas colunas dentro do container
        col1, col2 = st.columns([1, 3])

        with col1:
            st.image("logo_app.png", width=160)

        with col2:
            st.title("Biodiesel Properties Predictor")

    st.write(
        "<p style='text-align:justify;'>Fuels produced from microalgae offer a more environmentally sustainable alternative, given that they present a balanced ratio between the CO2 emitted during combustion and the subsequent absorption during biomass formation. However, there are numerous species of microalgae, not all of which are suitable for fuel production. With this perspective, the BPP emerges as an alternative to estimate the properties of a biodiesel produced from microalgae, receiving as input parameters the fatty acid profile of the strain under analysis </p>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.write("#### Enter the values for fatty acids:")

    # Criar inputs para os valores dos ácidos graxos
    inputs = {}
    for label, var_name in [
        ("Lauric acid (C12:0):", "entruc120"),
        ("Myristic acid (C14:0):", "entruc140"),
        ("Pentadecanoic acid (C15:0):", "entruc150"),
        ("Palmitic (C16:0):", "entruc160"),
        ("Palmitoleic (C16:1):", "entruc161"),
        ("Heptadecanoic (C17:0):", "entruc170"),
        ("Stearic acid (C18:0):", "entruc180"),
        ("Oleic acid (C18:1):", "entruc181"),
        ("Linoleic acid (C18:2):", "entruc182"),
        ("Linolenic acid (C18:3):", "entruc183"),
    ]:
        inputs[var_name] = st.number_input(label, value=0.0)

    if st.button("Predict"):
        input_values = [
            inputs["entruc120"],
            inputs["entruc140"],
            inputs["entruc150"],
            inputs["entruc160"],
            inputs["entruc161"],
            inputs["entruc170"],
            inputs["entruc180"],
            inputs["entruc181"],
            inputs["entruc182"],
            inputs["entruc183"],
        ]

        # Realize a previsão usando os modelos carregados
        predicted_cn = models["model_cn"].predict([input_values])[0]
        predicted_densidade = models["model_densidade"].predict([input_values])[0]
        predicted_hhv = models["model_hhv"].predict([input_values])[0]
        predicted_os = models["model_os"].predict([input_values])[0]
        predicted_iv = models["model_iv"].predict([input_values])[0]
        predicted_fscl = models["model_fscl"].predict([input_values])[0]
        predicted_cfpp = models["model_cfpp"].predict([input_values])[0]
        predicted_sv = models["model_sv"].predict([input_values])[0]
        predicted_lhv = models["model_lhv"].predict([input_values])[0]
        predicted_viscosidade = models["model_viscosidade"].predict([input_values])[0]

        # Exibir os resultados das previsões
        st.write("### Predicted Properties:")
        st.write(f"Cetan Number: {predicted_cn:.2f}")
        st.write(f"Density (g/cm^3): {predicted_densidade:.2f}")
        st.write(f"Higher Heating Value (MJ/kg): {predicted_hhv:.2f}")
        st.write(f"Oxidative Stability (h): {predicted_os:.2f}")
        st.write(f"Iode Value: {predicted_iv:.2f}")
        st.write(f"Long Chain Satured Factor: {predicted_fscl:.2f}")
        st.write(f"Cold Filter Plugin Point (°C): {predicted_cfpp:.2f}")
        st.write(f"Saponification Value: {predicted_sv:.2f}")
        st.write(f"Lower Heating Value (MJ/kg): {predicted_lhv:.2f}")
        st.write(f"Viscosity (mm^2/s): {predicted_viscosidade:.2f}")


# Executar a função principal
if __name__ == "__main__":
    app()