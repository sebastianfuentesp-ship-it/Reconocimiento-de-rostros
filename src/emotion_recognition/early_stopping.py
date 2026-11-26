"""Parada temprana compartida por A/B/C/D, basada en validación.

En las cuatro recetas actuales mode='min': se compara val_loss con su menor
valor histórico, no solo con la época anterior. Ocho épocas sin bajar ese mínimo
detienen el entrenamiento. El entrenamiento guarda el checkpoint cuando mejora.
"""

import math
from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    """Contar épocas consecutivas sin superar el mejor valor histórico.

    Con patience=8 y min_epochs=0 se cuenta desde el comienzo. Un periodo inicial
    opcional aplaza la parada, pero registra igualmente el mejor valor.
    patience=0 desactiva la parada; sigue registrando las mejoras.
    """

    patience: int = 8
    min_epochs: int = 0
    mode: str = "max"
    best_score: float = field(default=-math.inf, init=False)
    stale_epochs: int = field(default=0, init=False)
    stopped_epoch: int | None = field(default=None, init=False)

    def __post_init__(self):
        if self.mode not in {"min", "max"}:
            raise ValueError("mode debe ser min o max.")
        self.best_score = math.inf if self.mode == "min" else -math.inf
        for name in ("patience", "min_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} debe ser un entero no negativo.")

    def update(self, score: float, epoch: int) -> bool:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError("epoch debe ser un entero positivo.")
        if isinstance(score, bool) or not math.isfinite(score):
            raise ValueError("La métrica de validación debe ser finita.")
        if self.stopped_epoch is not None:
            return True
        # 1. Comparar con el mejor resultado de TODAS las épocas anteriores.
        improved = score < self.best_score if self.mode == "min" else score > self.best_score
        if improved:
            self.best_score = float(score)
        # 2. Una mejora reinicia el contador; un empate no es una disminución.
        if not self.patience or epoch <= self.min_epochs or improved:
            self.stale_epochs = 0
            return False
        # 3. Detener al agotar la paciencia. Nunca se compara la pérdida de train.
        self.stale_epochs += 1
        if self.stale_epochs >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False
