# Metodología de experimentos (sin leakage)

## Regla de oro
| Conjunto | Para qué se usa | Cuántas veces |
|---|---|---|
| **train** | Ajustar los pesos del modelo | Siempre |
| **val** | Comparar experimentos y elegir el mejor (y early stopping) | Todas las que haga falta |
| **test** (`dataset/juliaca/test`) | Medir el modelo FINAL ya elegido | **Una sola vez** |

Si se elige el modelo mirando el test, el test deja de ser una medida honesta (leakage por selección).
Cada uso de `juliaca/test` queda anotado en `reportes/registro_test.csv`.

## Métrica de selección
**F1 macro en validación**: promedia el F1 de cada clase por igual, así una clase con pocas fotos
(peligrosos, sin_residuo) pesa lo mismo que una con muchas. En empate se mira top-1.

## Diseño: un factor a la vez
Cada experimento cambia **una sola cosa** respecto a la referencia, para poder explicar su efecto.

**Etapa 1** (datos públicos; val = `publico/val`):
| Experimento | Cambio | Hipótesis |
|---|---|---|
| e1_base | yolo26s-cls, 224 px | Referencia |
| e1_drop02 | dropout 0.2 | Reduce el sobreajuste (loss_train ≪ loss_val) |
| e1_s320 | 320 px | Más detalle (textura del material) |
| e1_m224 | yolo26m-cls | Más capacidad |

**Etapa 2** (fine-tuning con Juliaca; val = `juliaca/val`), partiendo del ganador de la etapa 1:
| Experimento | Cambio | Hipótesis |
|---|---|---|
| e2_base | lr 1e-4, freeze 6, replay 1 | Referencia |
| e2_lr3e-5 / e2_lr3e-4 | learning rate | Muy bajo: aprende poco; muy alto: olvida la etapa 1 |
| e2_freeze0 / e2_freeze9 | capas congeladas | Con pocas fotos, congelar más evita memorizar |
| e2_replay0 / e2_replay2 | repaso de datos públicos | Sin repaso puede olvidar; con mucho, se adapta menos a Juliaca |
| e2_solo_jul | sin etapa 1 (desde ImageNet) | Mide cuánto aporta la etapa 1 |

## Comandos
```bash
python scripts/experimentar.py --listar                      # ver experimentos y resultados
python scripts/experimentar.py --etapa 1                     # corre todos los de la etapa 1
python scripts/experimentar.py --etapa 2 --pesos modelos/experimentos/<ganador_etapa1>.pt
python scripts/evaluar.py --pesos <ganador_etapa1>.pt <ganador_etapa2>.pt --split test   # UNA vez, al final
```
Resultados: `reportes/experimentos.csv` (tabla para el informe) y `reportes/experimentos/*.json`.
Curvas de cada entrenamiento: `runs/etapaN_<experimento>/results.png`.

## Qué reportar en el informe
1. Tabla de experimentos con F1 macro y top-1 en val, y el ganador de cada etapa.
2. Resultado en `juliaca/test` del ganador de la etapa 1 y del de la etapa 2 (mejora por datos locales).
3. Matriz de confusión y F1 por clase del modelo final: qué residuos confunde y por qué.
4. Limitaciones: la partición pública venía aleatoria por foto (se quitaron casi-duplicados por hash
   perceptual, pero puede quedar correlación); por eso la métrica válida es `juliaca/test`, separado por objeto.
5. Que el data augmentation (giros, color, RandAugment) transforma fotos reales al entrenar; no se generaron datos.
