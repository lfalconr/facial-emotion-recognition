# Facial Emotion recognition on static images and webcam

<p>Este projeto faz parte do trabalho T3 da disciplina Computação Afetiva realizada na Unicamp durante o 2°Semestre do 2019, oferecida pela Professora Paula Dornhofer Paro Costa. </p>
<p>O projeto está baseado no tutorial Emotion Recognition using Facial Landmarks, Python, DLib and OpenCV (http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/)</p>

<h3>Requisitos</h3>
<p>Para a execução do trabalho é necessário ter instalado:</p>
<ul>
  <li>Python 3</li>
  <li>Opencv</li>
  <li>SKLearn</li>
  <li>Dlib (instruções para compilá-lo http://dlib.net/compile.html)</li>
  <li>Matplotlib</li>
  <li>Pandas</li>
  <li>Seaborn</li>
   <li>Visual studio</li>
 </ul>
 
 <h3>Dataset</h3>
 <p>Foi usado o dataset Cohn-Kanade, versão 2 (CK +) para treinar e validar o modelo. O banco de dados consiste em um grupo de sequências de imagens de pessoas realizando uma expressão facial correspondente a uma das emoções universais. Os dados são rotulados. O dataset pode ser solicitado aqui: http://www.consortium.ri.cmu.edu/ckagree/</p>
 
 <h3>Preparação dos Dados</h3>
 <p>Extraia o dataset e coloque todas as pastas que contêm os arquivos txt (S005, S010 etc.) em uma pasta chamada <b>"source_emotion"</b> dentro da pasta existente <b>"assets"</b>. Coloque as pastas que contêm as imagens em uma pasta chamada <b>"source_images"</b> dentro de <b>"assets"</b>.</p>
<p>Coloque <b>shape_predictor_68_face_landmarks.dat</b> dentro da pasta <b>"assets"</b>. Pode ser obtido aqui: https://github.com/davisking/dlib-models</p>
 
 <h3>Composição do Projeto</h3>
 <p>O projeto é composto por 4 passos:</p> 
 <p>O primeiro deles <b>A6-1</b> demonstra como o detector de landmarks do dlib funciona usando imagens estáticas ou em tempo real usando a webcam.</p>
 <p>A segunda etapa corresponde à preparação do dataset. Depois de preparar os dados mencionados acima, o arquivo <b>A6-2</b> pode ser executado para organizar o conjunto de dados.</p>
 <p>Na terceira das etapas <b>A6-3</b>, o modelo é criado. É usada a técnica Support Vector Machine (SVM) do Aprendizado de Máquina. É realizada uma comparação dos resultados usando kernels lineares, polinomiais, sigmóides e rbf. O modelo com kernel linear é selecionado porque ele tem maior accuracy neste problema.</p>
 <p>Como último passo <b>A6-4</b>,  é realizado o teste de detecção de emoções em tempo real usando a webcam.</p>







