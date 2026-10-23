"""Verifica y compara versiones entre PC y laptop.

En la PC (después de instalar):   python scripts/verificar_entorno.py --guardar entorno_referencia.json
En la laptop:                     python scripts/verificar_entorno.py --comparar entorno_referencia.json
"""
import argparse
import json
import platform
import sys


def recoger():
    info = {"python": platform.python_version(), "so": platform.platform()}
    try:
        import torch
        info.update({
            "torch": torch.__version__,
            "cuda_torch": torch.version.cuda,
            "cuda_disponible": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
        })
    except ImportError:
        info["torch"] = None
    for mod in ["torchvision", "ultralytics", "cv2", "numpy", "PIL"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except ImportError:
            info[mod] = None
    return info


IMPORTAN = ["python", "torch", "torchvision", "ultralytics", "cv2", "numpy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guardar")
    ap.add_argument("--comparar")
    args = ap.parse_args()

    info = recoger()
    for k, v in info.items():
        print(f"{k:16s} {v}")
    if not info.get("cuda_disponible"):
        print("\n[ERROR] torch no ve la GPU. Reinstala torch con CUDA desde el selector de pytorch.org.")

    if args.guardar:
        with open(args.guardar, "w") as f:
            json.dump(info, f, indent=2)
        print(f"\nGuardado en {args.guardar}")

    if args.comparar:
        ref = json.load(open(args.comparar))
        print(f"\nComparando con {args.comparar}:")
        dif = 0
        for k in IMPORTAN:
            a = str(ref.get(k)).split("+")[0]
            b = str(info.get(k)).split("+")[0]
            if k == "python":
                a, b = ".".join(a.split(".")[:2]), ".".join(b.split(".")[:2])
            estado = "OK" if a == b else "DIFERENTE"
            dif += estado != "OK"
            print(f"  {k:14s} ref={a:14s} aqui={b:14s} {estado}")
        print("\nTodo coincide." if dif == 0 else f"\n{dif} diferencias: pip install -r requirements-lock.txt")
        sys.exit(1 if dif else 0)


if __name__ == "__main__":
    main()
