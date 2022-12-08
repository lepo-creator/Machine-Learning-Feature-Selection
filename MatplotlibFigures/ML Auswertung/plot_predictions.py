def plot_predictions(model, inputs, labels, x, export): #input: Vorhergesagte Schmelzpoolabmaße in x/y/z-Richtung , Simulierte Schmelzpoolabmaße in x/y/z-Richtung
    plt.figure(figsize=(16/2.54, 4.5/2.54), dpi=250)
    c = ['#000099', '#D1B926']; v = 0
    for j,k in zip(inputs, labels): 
        input_prediction = scaler(j)
        prediction = pd.DataFrame(model.predict(input_prediction), index = k.index, columns = k.columns)   
        u = 1
        for i in x:
            plt.subplot(1,len(x),u)
            plt.grid(alpha = 0.4, linestyle='dotted', linewidth=0.7)
            plt.scatter(k[i], prediction[i], marker='o', s = 4, color = c[v], label="Vorhergesagte ")
            plt.plot(range(200), color = '#C74D47', label="Data=Prediction")
           
            plt.xlim(k[i].min()*0.997, k[i].max()*1.003)
            plt.ylim(k[i].min()*0.997, k[i].max()*1.003)
            plt.xticks(np.linspace(round(min(k[i]),3), round(max(k[i]),3), 3))
            plt.yticks(np.linspace(round(min(k[i]),3), round(max(k[i]),3), 3))
            u += 1
        v=+1
    plt.legend = []
    plt.tight_layout()
    if export == True:
        name = str(model).split('(')[0]
        plt.savefig('PlotExport/Vorhersagegenauigkeit_' + name + '.svg',dpi=250)
    plt.show()
    return

    # Darstellung der Vorhersagen für die unregulierten Modelle (Plotbreite muss auf 16/2.54 angepasst werden)
#for i in [lin_reg, dt_reg, rf_reg, mlp_reg]:
#    plot_predictions(i, [training_input, testing_input], [training_labels, testing_labels], ["x","y","z"], False)