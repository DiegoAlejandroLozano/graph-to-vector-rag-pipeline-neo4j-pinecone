from busqueda_graphrag_autonomo import GraphRAGAutonomo

if __name__ == "__main__":
    app = GraphRAGAutonomo()

    try:
        # Ejemplo 1: Pregunta directa por entidad
        res1 = app.ejecutar_pipeline("¿Quién creó el Django REST Framework y para qué sirve?")
        print("\n=== RESPUESTA 1 ===")
        print(res1)

        print("\n" + "="*50 + "\n")

        # Ejemplo 2: Pregunta que busca por propiedad ('Tom Christie' o 'Swagger')
        res2 = app.ejecutar_pipeline("¿Qué herramientas o versiones están relacionadas con Swagger?")
        print("\n=== RESPUESTA 2 ===")
        print(res2)
    finally:
        app.cerrar()