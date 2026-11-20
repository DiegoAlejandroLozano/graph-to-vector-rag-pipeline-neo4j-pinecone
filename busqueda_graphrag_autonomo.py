from typing                 import List, Dict, Any
from dotenv                 import dotenv_values
from pydantic               import BaseModel, Field
from pinecone               import Pinecone
from neo4j                  import GraphDatabase
from langchain_openai       import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------------------
# ESQUEMA PYDANTIC PARA EXTRAER FILTROS DESDE LA CONSULTA
# -------------------------------------------------------------
class FiltrosConsultaGraphRAG(BaseModel):
    terminos_clave: List[str] = Field(
        description="Lista de nombres de entidades, tecnologías, conceptos o propiedades a buscar en el grafo."
    )

class GraphRAGAutonomo:

    """Pipeline GraphRAG Autónomo con Extracción por Ontología."""

    def __init__(self):
        config          = dotenv_values(dotenv_path=".env")
        self.openai_key = config.get("OPENAI_API_KEY")

        # 1. Neo4j
        neo4j_uri      = config.get("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user     = config.get("NEO4J_USER", "neo4j")
        neo4j_password = config.get("NEO4J_PASSWORD", "password")
        self.driver    = GraphDatabase.driver(
            uri=neo4j_uri,
            auth=(
                neo4j_user,
                neo4j_password
            )
        )

        # 2. Pinecone
        pinecone_api_key = config.get("PINECONE_API_KEY")
        index_name       = config.get("PINECONE_INDEX_NAME", "knowledge-graph")
        pc               = Pinecone(api_key=pinecone_api_key)
        self.index       = pc.Index(name=index_name)

        # 3. Modelos
        self.embeddings = OpenAIEmbeddings(
            model   = "text-embedding-3-small",
            api_key = self.openai_key
        )

        self.llm = ChatOpenAI(
            model       = "gpt-4o-mini",
            temperature = 0.0,
            api_key     = self.openai_key
        )

        # LLM con salida estructurada para extraer filtros
        self.extractor_filtros = self.llm.with_structured_output(schema=FiltrosConsultaGraphRAG)

        # Extraer la ontología real directamente desde la base de datos
        self.ontologia_real = self._obtener_esquema_ontologico_dinamico()

        self.system_prompt_enrutador = (
            "Eres el motor de enrutamiento y generación de filtros para un sistema GraphRAG.\n"
            "Tu objetivo es analizar la pregunta del usuario y extraer los términos de búsqueda "
            "basándote EXCLUSIVAMENTE en la ontología activa de la base de datos.\n\n"
            f"{self.ontologia_real}\n\n"
            "REGLAS:\n"
            "1. Extrae únicamente entidades, tecnologías, creadores o versiones que coincidan con la pregunta."
        )

    def _obtener_esquema_ontologico_dinamico(self) -> str:
        """
        Inspecciona la base de datos Neo4j en tiempo real para construir 
        el mapa exacto de etiquetas, relaciones y propiedades disponibles.
        """
        with self.driver.session() as session:
            # 1. Obtener todas las etiquetas de nodos (labels)
            res_labels = session.run("CALL db.labels()")
            labels     = [r[0] for r in res_labels]

            # 2. Obtener todos los tipos de relaciones
            res_rels = session.run("CALL db.relationshipTypes()")
            rels     = [r[0] for r in res_rels]

            # 3. Obtener una muestra de claves de propiedades usadas en nodos
            res_props_nodos = session.run("""
                MATCH (n)
                UNWIND keys(n) AS k
                RETURN DISTINCT k
            """)
            props_nodos = [r[0] for r in res_props_nodos]

            # 4. Obtener claves de propiedades usadas en relaciones
            res_props_rels = session.run("""
                MATCH ()-[r]->()
                UNWIND keys(r) AS k
                RETURN DISTINCT k
            """)
            props_rels = [r[0] for r in res_props_rels]

            # Formatear como un bloque de texto claro para el Prompt
            esquema_texto = (
                f"ESQUEMA REAL EXTRAÍDO DE NEO4J:\n"
                f"- Etiquetas de Nodos Disponibles: {', '.join(labels)}\n"
                f"- Tipos de Relaciones Existentes: {', '.join(rels)}\n"
                f"- Claves de Propiedades Frecuentes: {', '.join(props_nodos)}\n"
                f"- Claves de Propiedades de Relaciones: {", ".join(props_rels)}"
            )

            return esquema_texto

    def _limpiar_props(self, props: Dict[str, Any]) -> str:
        if not props:
            return ""
        # Excluimos id y nombre para evitar redundancia
        props_filtradas = {k: v for k, v in props.items() if k not in ["id", "nombre"]}
        return f" {props_filtradas}" if props_filtradas else ""

    def _obtener_subgrafo_y_node_ids(self, filtros: FiltrosConsultaGraphRAG) -> Dict[str, Any]:
        """
        Paso 2: Búsqueda transversal en Neo4j evaluando tanto el campo `nombre` 
        como todas las propiedades de cada nodo (`versión`, `creador`, `anterior`, etc.).
        """
        node_ids_candidatos = set()
        contexto_grafo      = []
        query_cypher        = """
        MATCH (n:Entidad)-[r]-(m:Entidad)
        WHERE (
            ANY(t IN $terminos WHERE toLower(n.nombre) CONTAINS toLower(t))
            OR ANY(t IN $terminos WHERE ANY(key IN keys(n) WHERE toLower(toString(n[key])) CONTAINS toLower(t)))
            OR ANY(t IN $terminos WHERE ANY(key IN keys(r) WHERE toLower(toString(r[key])) CONTAINS toLower(t)))
            OR ANY(t IN $terminos WHERE toLower(m.nombre) CONTAINS toLower(t))
            OR ANY(t IN $terminos WHERE ANY(key IN keys(m) WHERE toLower(toString(m[key])) CONTAINS toLower(t)))
        )
        RETURN
            n.id             AS nodo_origen_id,
            n.nombre         AS nodo_origen_nombre,
            properties(n)    AS nodo_origen_props,
            type(r)          AS relacion,
            properties(r)    AS relacion_props,
            m.id             AS nodo_destino_id,
            m.nombre         AS nodo_destino_nombre,
            properties(m)    AS nodo_destino_props,
            startNode(r) = n AS es_origen_real 
        LIMIT 50
        """

        with self.driver.session() as session:
            result = session.run(
                query    = query_cypher, 
                terminos = filtros.terminos_clave,
            )
            for record in result:
                origen_id      = record["nodo_origen_id"]
                origen_nombre  = record["nodo_origen_nombre"]
                origen_props   = record["nodo_origen_props"]
                rel            = record["relacion"]
                rel_props      = record["relacion_props"]
                destino_id     = record["nodo_destino_id"]
                destino_nombre = record["nodo_destino_nombre"]
                destino_props  = record["nodo_destino_props"]
                es_origen_real = record["es_origen_real"]

                if origen_id:
                    node_ids_candidatos.add(origen_id)
                if destino_id:
                    node_ids_candidatos.add(destino_id)

                # Formateo de propiedades limpias
                props_origen_str  = self._limpiar_props(props=origen_props)
                props_destino_str = self._limpiar_props(props=destino_props)
                props_rel_str     = self._limpiar_props(props=rel_props) if rel_props else ""
 
                if rel and destino_nombre:
                    rel_con_props = f"-[{rel}{props_rel_str}]->" if props_rel_str else f"-[{rel}]->"
                    if es_origen_real:
                        contexto_grafo.append(
                            f"({origen_nombre}{props_origen_str}){rel_con_props}({destino_nombre}{props_destino_str})"
                        )
                    else:
                        contexto_grafo.append(
                            f"({destino_nombre}{props_destino_str}){rel_con_props}({origen_nombre}{props_origen_str})"
                        )
                else:
                    contexto_grafo.append(f"Entidad: {origen_nombre} | Atributos: {origen_props}")

        return {
            "node_ids"       : list(node_ids_candidatos),
            "contexto_texto" : list(set(contexto_grafo))
        }

    def _extractor_filtros_desde_pregunta(self, pregunta: str) -> FiltrosConsultaGraphRAG:
        """Paso 1: Extrae entidades usando el LLM con ontología inyectada"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt_enrutador),
            ("human", "{pregunta}")
        ])
        chain = prompt | self.extractor_filtros
        return chain.invoke({'pregunta':pregunta})

    def _bucar_en_pinecone(self, node_ids: List[str]) -> List[str]:
        """Paso 3: Búsqueda vectorial filtrada por los node_ids validados por Neo4j"""

        if not node_ids:
            return []

        # Recupera directamente los registros cuyas claves coincien con los nodos_ids 
        res_pinecone  = self.index.fetch(ids=node_ids)
        textos_unicos = set()
        for _, vector_data in res_pinecone.vectors.items():
            metadata = vector_data.metadata
            if metadata:
                texto = metadata.get("text")
                if texto:
                    textos_unicos.add(texto.strip())
        textos_recuperados = [
            f"\n--- Contexto de Origen {i:02d} ---\n{txt}"
            for i, txt in enumerate(textos_unicos, 1)
        ]
        return textos_recuperados

    def ejecutar_pipeline(self, pregunta_usuario:str) -> str:
        print(f"\n[1. Ontologia/Filtros] Analizando pregunta: '{pregunta_usuario}'")
        filtros = self._extractor_filtros_desde_pregunta(pregunta_usuario)
        print(f"-> Términos: {filtros.terminos_clave}")

        if not filtros.terminos_clave:
            return "No se encontraron entidades ni conceptos clave para consultar en la base de conocimientos"

        # Extraer subgrafo de Neo4j
        datos_grafo    = self._obtener_subgrafo_y_node_ids(filtros=filtros)
        node_ids       = datos_grafo["node_ids"]
        contexto_grafo = datos_grafo["contexto_texto"]
        print(f"[2. Neo4j Subgrafo] Nodos candidatos recuperados: {len(node_ids)}")

        # Paso 3: Recuperar fragmentos vectoriales en Pinecone
        contexto_vectorial = self._bucar_en_pinecone(node_ids=node_ids)
        print(f"[3. Pinecone Vectorial] Registros recuperados: {len(contexto_vectorial)}")

        # Paso 4: Sintesis final con LLM
        prompt_sintesis = ChatPromptTemplate.from_messages([
            ("system", (
                "Eres un asistente experto en bases de conocimiento.\n"
                "Responde a la pregunta del usuario utilizando EXCLUSIVAMENTE el contexto proveído a continuación.\n"
                "Si la información no es suficiente para responder, indícalo claramente sin inventar datos.\n\n"
                "1. ESTRUCTURA Y RELACIONES DEL GRAFO:\n"
                "{contexto_grafo}\n\n"
                "2. FRAGMENTOS DE TEXTO RECUPERADOS:\n"
                "{contexto_vectorial}"
            )),
            ("human", "{pregunta}")
        ])

        # Preparar los valores del contexto
        valores_prompt = {
            "contexto_grafo"     : "\n".join(contexto_grafo) if contexto_grafo else "Sin relaciones",
            "contexto_vectorial" : "\n".join(contexto_vectorial) if contexto_vectorial else "Sin propiedades adicionales.",
            "pregunta"           : pregunta_usuario
        }

        # --- IMPRIMIR EL SYSTEM PROMPT COMPLETO FORMATEADO ---
        # prompt_formateado = prompt_sintesis.format(**valores_prompt)
        # print("\n" + "="*80)
        # print("SYSTEM PROMPT ENVIADO AL LLM DE SÍNTESIS:")
        # print("="*80)
        # print(prompt_formateado)
        # print("="*80 + "\n")

        cadena_sintesis = prompt_sintesis | self.llm
        respuesta = cadena_sintesis.invoke(valores_prompt)

        return respuesta.content

    def cerrar(self):
        """Cierra el driver de Neo4j al finalizar."""
        self.driver.close()