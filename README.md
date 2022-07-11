# Módulo de Python para el Trabajo de Fin de Grado de Martín Alcalde Martínez (uo269426@uniovi.es)

Es importante notar que no se incluyen los datos usados ya que son archivos demasiado pesados.



## Estructura del paquete:
- **DQM** (*Data Quality Management*):<br>

    - **classes**:
        - **Data**: Clase que sirve para leer, limpiar y ordenar los datos que se leen de los archivos csv en los que se encuentran los datos. Permite también leer y guardar los objetos creados para su uso posterior.

        - **Model**: Clase que sirve para el manejo de los modelos numéricos de forma más cómoda. Permite el entrenamiento y la evaluación, así como la representación gráfica de varias cantidades. Al igual que Data, se puede guardar y leer para eliminar la necesidad de entrenar el modelo cada vez que se quiera usar.

        - **Metric**: Clase que usan de base todas las métricas a través de la _inheritance_ de Python. Permite evaluar la métrica dada a partir de los datos reales y la reconstrucción, así como representar gráficamente una serie de distribuciones de interés de los valores de la métrica.

            - **MSE**: Métrica que se va a emplear en este trabajo. Es el error cuadrático medio de la reconstrucción con respecto a los datos reales. Para el etiquetado, exigiremos una cota de este error a través de un objeto Filter.

        - **Filter**: Clase que usan de base todos los filtros que se han programado. Devuelve etiquetas True o False según si se cumple la condición del filtro.

            - **Entries**: Filtro empleado para poner una cota al número mínimo de entradas que deben de tener las LS para ser consideradas.

            - **MinMax**: Filtro empleado para establecer una cota superior e inferior a un valor y devolver True para todos los valores contenidos entre esas cotas.<br>_Nota: Se sospecha que esta clase presenta algún tipo de error y las etiquetas obtenidas a través de ella podrían no ser correctas._

            - **Training**: Filtro en el cual se especifican la fracción de datos buenos y datos malos que se desea tomar para el entrenamiento y se cogen LS aleatorias hasta que se llegue a la proporción indicada. Se usa en conjunto al filtro Validation para separar un objeto Data en dos objetos Data, formando los conjuntos de entrenamiento y validación.

            - **Validation**: Toma las etiquetas de un filtro Training y las invierte. Se usa en conjunto al filtro Training para separar un objeto Data en dos objetos Data, formando los conjuntos de entrenamiento y validación.
        
        - **HyperData**: Agrupación de cuatro objetos Data (uno por cada observable). Es usado para proporcionar los datos de entrenamiento o validación de un objeto HyperModel.

        - **HyperModel**: Agrupación de cuatro objetos Model (uno por cada observable). Es usado para obtener el modelo completo que tiene en cuenta las etiquetas de los cuatro observables y las junta aplicando una conjunción lógica (intersección de los conjuntos).


    - **utils**:


        - **data**: Aquí se incluyen muchos de las variables auxiliares como la lista de los observables o de los archivos csv, o incluso la configuración de los entrenamientos y evaluaciones.

        - **dataframes**: Funciones usadas para el tratamiento de las estructuras de datos.

        - **logging**: Funciones que envuelven funciones del módulo con el mismo nombre para un uso más cómodo.

        - **plots**: Funciones usadas para la representación gráfica.

        - **threading**: Funciones que envuelven funciones del módulo con el mismo nombre para ampliar sus utilidades.


