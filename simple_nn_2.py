# -*- coding: utf-8 -*-
import time

import numpy
# сигмоид expit()
import scipy.special

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from fpdf import FPDF
import base64


# global vars
my_neural_network = None

# Задание архитектуры сети:
# количество входных, скрытых и выходных узлов
input_nodes = 784
output_nodes = 10

allow_export_as_pdf = False

# количкство скрытых слоёв
hidden_layers_n = st.slider('number of hidden layers', 1, 5, 1,
                            help="Set here the number of hidden layers of your network")

hidden_layers_lengths = list()
for i in range(hidden_layers_n):
    hidden_layers_lengths.append(int(st.number_input('nodes hidden in layer ' + str(i+1),
                                                     value=20,
                                                     max_value=input_nodes,
                                                     min_value=output_nodes,
                                                     step=1,
                                                     key="number_input_"+str(i),
                                                     help="Set here the number of nodes in layer " + str(i+1) +
                                                          ". Value must be higher than number of outpot"
                                                          " nodes and less than one of input nodes.")))

# уровень обучения
learning_rate = 10**st.slider('log learning rate', -5.0, 0.0, -1.0, step=0.1,
                              help="Here is decimal logarithm of learning rate")

# Обучение нейронной сети
# количество эпох
epochs = st.slider('epochs number', 1, 100, 1, step=1,
                   help="Set here the number of epochs for the network training process.")

x_plt_data = list()
y_plt_data_test = list()
y_plt_data_train = list()
test_res_table = {
        "Epoch": [],
        "Performance on test data, %": []
    }


# описание класса нейронной сети
class NeuralNetwork:

    # инициализация нейронной сети
    def __init__(self, input_n, h_layers_lengths, out_nodes, l_rate):
        # задание количества узлов входного, скрытого и выходного слоя
        self.i_nodes = input_n
        self.h_nodes = h_layers_lengths
        self.o_nodes = out_nodes

        # связь весовых матриц, wih и who
        # вес внутри массива w_i_j, где связь идет из узла i в узел j
        # следующего слоя
        # w11 w21
        # w12 w22 и т д7
        self.hidden_layers = list()
        current_inodes = self.i_nodes
        for h_nodes_i in self.h_nodes:
            self.hidden_layers.append(numpy.random.normal(0.0, pow(current_inodes, -0.5),
                                                          (h_nodes_i, current_inodes)))
            current_inodes = h_nodes_i

        self.who = numpy.random.normal(0.0, pow(int(self.h_nodes[-1]), -0.5), (self.o_nodes, int(self.h_nodes[-1])))

        # уровень обучения
        self.lr = l_rate

        # функция активации - сигмоид
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

        # обучение нейронной сети

    def train(self, inputs_list, targets_list):
        # преобразование входного списка 2d массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_outputs_list = list()
        current_inputs = inputs
        for h_layer in self.hidden_layers:
            # вычисление сигналов на входе в скрытый слой
            hidden_inputs = numpy.dot(h_layer, current_inputs)
            # вычисление сигналов на выходе из скрытого слоя
            hidden_outputs = self.activation_function(hidden_inputs)
            hidden_outputs_list.append(hidden_outputs)
            current_inputs = hidden_inputs

        # вычисление сигналов на входе в выходной слой
        final_inputs = numpy.dot(self.who, hidden_outputs_list[-1])
        # вычисление сигналов на выходе из выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибка на выходе (целевое значение - рассчитанное)
        output_errors = targets - final_outputs
        # распространение ошибки по узлам скрытого слоя
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # пересчет весов между скрытым и выходным слоем
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs_list[-1]))

        for n_i, h_o in reversed(list(enumerate(hidden_outputs_list))):
            if n_i > 0:
                # пересчет весов между скрытыми слоями
                self.hidden_layers[n_i] += self.lr * numpy.dot((hidden_errors * h_o * (1.0 - h_o)),
                                                               numpy.transpose(hidden_outputs_list[n_i-1]))
                hidden_errors = numpy.dot(self.hidden_layers[n_i].T, hidden_errors)
        # пересчет весов между входным и скрытым слоем
        self.hidden_layers[0] += self.lr * numpy.dot((hidden_errors * hidden_outputs_list[0] *
                                                     (1.0 - hidden_outputs_list[0])),
                                                     numpy.transpose(inputs))

    # запрос к нейронной сети
    def query(self, inputs_list):
        # преобразование входного списка 2d массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_outputs = None
        for layer in self.hidden_layers:
            # вычисление сигналов на входе в скрытый слой
            hidden_inputs = numpy.dot(layer, inputs)
            # вычисление сигналов на выходе из скрытого слоя
            hidden_outputs = self.activation_function(hidden_inputs)
            inputs = hidden_outputs

        # вычисление сигналов на входе в выходной слой
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # вычисление сигналов на выходе из выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


def test(n, data_file="mnist_test.csv"):
    # Загрузка тестового набора данных
    test_data_file = open(data_file, 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # Тестирование нейронной сети

    # Создание пустого накопителя для оценки качества
    scorecard = []

    # итерирование по тестовому набору данных
    for record in test_data_list:
        # разделение записей по запятым ','
        all_values = record.split(',')
        # правильный ответ - в первой ячейке
        correct_label = int(all_values[0])
        # масштабирование и сдвиг исходных данных
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # получение ответа от нейронной сети
        outputs = n.query(inputs)
        # получение выхода
        label = numpy.argmax(outputs)
        # добавление в список единицы, если ответ совпал с целевым значением
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    scorecard_array = numpy.asarray(scorecard)
    print(data_file, scorecard_array.size)
    return scorecard_array.sum() / scorecard_array.size


def do_test(n, report_data=None):
    score = 100 * test(n, data_file="mnist_test.csv")
    print("performance on test data = ", score, " %")
    y_plt_data_test.append(score)
    st.write("performance on test data: ", score, " %")
    test_res_table["Performance on test data, %"].append(round(100*score)/100)
    if test_on_train_data:
        score = 100 * test(n, data_file="mnist_train.csv")
        print("performance on train data = ", score, " %")
        y_plt_data_train.append(score)
        st.write("performance on train data: ", score, " %")
        test_res_table["Performance on train data, %"].append(round(100*score)/100)
    if not test_after_each_epoch:
        if not show_table:
            report_data.append("Performance on test data: " + str(test_res_table["Performance on test data, %"][0])
                               + " %")
            if test_on_train_data:
                report_data.append("Performance on train data: " +
                                   str(test_res_table["Performance on train data, %"][0]) + " %")
            create_pgf_report(report_data)


if st.checkbox('Test after each epoch'):
    test_after_each_epoch = True
    if st.checkbox('Show table'):
        show_table = True
    else:
        show_table = False
else:
    test_after_each_epoch = False
    show_table = False


if st.checkbox('Test also on train data'):
    test_on_train_data = True
    test_res_table["Performance on train data, %"] = list()
else:
    test_on_train_data = False


def draw_plot(ax, test_on_train_data):
    line1, = ax.plot(x_plt_data, y_plt_data_test, label='performance on test data', color='red'),
    if test_on_train_data:
        line2,  = ax.plot(x_plt_data, y_plt_data_train, label='performance on train data', color='blue')
    plt.legend()
    st.pyplot(plt)


def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


def do_train(ep_n, output):
    global my_neural_network, x_plt_data, y_plt_data_test, y_plt_data_train
    report_data = list()
    report_data.append("Report about neural network training and performance")
    report_data.append("")
    report_data.append("Hyper parameters")
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch No', fontsize=15, color='blue')
    ax.set_ylabel('Performance, %', fontsize=15, color='blue')
    ax.set_title('Neural network performance test', fontsize=17)
    plt.grid(True)
    x_plt_data = list()
    y_plt_data_test = list()
    y_plt_data_train = list()
    # Загрузка тренировочного набора данных
    st.write("loading data...")
    start_loading_moment = time.time()
    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    st.write("data are loaded")
    st.write("loading time ", time.time() - start_loading_moment, " seconds")

    # создание экземпляра класса нейронной сети
    st.write("neural network creation...")
    start_n_moment = time.time()
    my_neural_network = NeuralNetwork(input_nodes,
                                      hidden_layers_lengths,
                                      output,
                                      learning_rate)
    st.write("neural network is created")
    st.write("nodes in input layer: ", my_neural_network.i_nodes)
    st.write("nodes in out layer: ", my_neural_network.o_nodes)
    st.write("nodes in hidden layers: ", hidden_layers_lengths)
    st.write("creation time ", time.time() - start_n_moment, " seconds")
    st.write("starting the train process with following hyper parameters:")
    st.write("total epochs number: ", ep_n)
    st.write("leaning rate: ", learning_rate)
    report_data.append("Nodes in input layer: " + str(my_neural_network.i_nodes))
    report_data.append("Nodes in out layer: " + str(my_neural_network.o_nodes))
    if len(hidden_layers_lengths) > 1:
        report_data.append("Number of hidden layers: " + str(len(hidden_layers_lengths)))
        report_data.append("Nodes in hidden layers: " + ", ".join([str(j) for j in hidden_layers_lengths]))
    else:
        report_data.append("Nodes in hidden layer: " + " ".join([str(j) for j in hidden_layers_lengths]))
    report_data.append(("Total epochs number: " + str(ep_n)))
    report_data.append(("Leaning rate: " + str(learning_rate)))

    start_train_moment = time.time()
    for e in range(ep_n):
        # итерирование по всем записям обучающего набора
        x_plt_data.append(e+1)
        st.write("epoch No: ", e+1)
        test_res_table["Epoch"].append(e+1)
        start_epoch_moment = time.time()
        for record in training_data_list:
            # разделение записей по запятым ','
            all_values = record.split(',')
            # масштабирование и сдвиг исходных данных
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # создание целевых  выходов
            targets = numpy.zeros(output) + 0.01
            # элемент all_values[0] является целевым для этой записи
            targets[int(all_values[0])] = 0.99
            my_neural_network.train(inputs, targets)
        st.write("epoch time: ", time.time() - start_epoch_moment, " seconds")
        if test_after_each_epoch:
            do_test(my_neural_network)

    st.write("the train process is compete")
    st.write("total train time = ", time.time() - start_train_moment, " seconds")
    if test_after_each_epoch and len(x_plt_data) > 1:
        draw_plot(ax, test_on_train_data)
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = Image.fromarray(numpy.asarray(canvas.buffer_rgba()))
        report_data.append(img)
        if show_table:
            st.table(test_res_table)
            report_data.append("")
            report_data.append("Neural network performance change through the epochs")
            report_data.append(test_res_table)
    if not test_after_each_epoch:
        st.button("TEST TRAINED NETWORK", on_click=do_test, args=(my_neural_network, report_data, ))
    else:
        create_pgf_report(report_data)


def create_pgf_report(input):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    for i, item in enumerate(input):
        if i == 0:
            pdf.set_font('Arial', 'B', 16)
        else:
            pdf.set_font('Arial', style = '', size = 12)
        if isinstance(item, str):
            pdf.cell(40, 10, item)
        elif isinstance(item, list):
            for row in item:
                # получаем данные колонки таблицы
                for datum in row:
                    # выводим строку с колонками
                    pdf.multi_cell(pdf.epw / 4, pdf.font_size * 2, str(datum), border=1, ln=3,
                                   max_line_height=pdf.font_size)
        elif isinstance(item, dict):
            pdf.set_font('Arial', 'B', 10)
            len_table = 0
            for key in item:
                len_table = len(item[key])
                pdf.multi_cell(pdf.epw / len(item), pdf.font_size * 2, str(key), border=1, ln=3,
                               max_line_height=pdf.font_size)
            pdf.ln(pdf.font_size * 2)
            rows = list()
            for j in range(len_table):
                line = list()
                for n, key in enumerate(item):
                    line.append(item[key][j])
                rows.append(line)

            pdf.set_font('Arial', style='', size=10)
            for line in rows:
                for datum in line:
                    pdf.multi_cell(pdf.epw / len(line), pdf.font_size * 2, str(datum), border=1,
                                   ln=3, max_line_height=pdf.font_size)
                pdf.ln(pdf.font_size * 2)
        else:
            print(item)
            print(type(item))
            pdf.image(item, w=pdf.epw)

        pdf.ln(pdf.font_size * 2)

    pdf.output("report.pdf")
    with open('report.pdf', 'rb') as f:
        st.download_button('Download report', f, file_name='report.pdf')



st.button("TRAIN", on_click=do_train, args=(epochs, output_nodes))


