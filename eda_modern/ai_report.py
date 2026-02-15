"""Assistant de génération de rapport IA via Hugging Face."""

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import pandas as pd


class HuggingFaceReportAssistant:
    """Génère une interprétation IA des résultats EDA via Hugging Face Inference API."""

    MODEL_PROFILES = {
        # Priorité: génération de rapport structuré
        "report_generation": {
            "chat_model": "openbmb/AgentCPM-Report:hf-inference",
            "fallback_chat_models": [
                "HuggingFaceTB/SmolLM3-3B:hf-inference",
                "Qwen/Qwen2.5-7B-Instruct:novita",
                "mistralai/Mistral-7B-Instruct-v0.3:featherless-ai",
            ],
            "api_url": "https://router.huggingface.co/hf-inference/models/openbmb/AgentCPM-Report",
            "fallback_model_ids": [
                "openbmb/AgentCPM-Report",
                "HuggingFaceTB/SmolLM3-3B",
                "Qwen/Qwen2.5-7B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
            ],
        },
        # Priorité: très long contexte / synthèse documentaire
        "long_context_research": {
            "chat_model": "moonshotai/Kimi-K2-Instruct-0905:hf-inference",
            "fallback_chat_models": [
                "Qwen/Qwen2.5-7B-Instruct:novita",
                "meta-llama/Llama-3.1-8B-Instruct:nebius",
                "HuggingFaceTB/SmolLM3-3B:hf-inference",
            ],
            "api_url": "https://router.huggingface.co/hf-inference/models/moonshotai/Kimi-K2-Instruct-0905",
            "fallback_model_ids": [
                "moonshotai/Kimi-K2-Instruct-0905",
                "Qwen/Qwen2.5-7B-Instruct",
                "meta-llama/Llama-3.1-8B-Instruct",
            ],
        },
        # Priorité: modèle lourd pour rédaction institutionnelle
        "enterprise_heavy": {
            "chat_model": "tiiuae/Falcon-180B-Chat:hf-inference",
            "fallback_chat_models": [
                "meta-llama/Llama-3.1-8B-Instruct:nebius",
                "Qwen/Qwen2.5-7B-Instruct:novita",
            ],
            "api_url": "https://router.huggingface.co/hf-inference/models/tiiuae/Falcon-180B-Chat",
            "fallback_model_ids": [
                "tiiuae/Falcon-180B-Chat",
                "meta-llama/Llama-3.1-8B-Instruct",
            ],
        },
        # Priorité: pipeline résumé puis génération
        "summarization_first": {
            "chat_model": "HuggingFaceTB/SmolLM3-3B:hf-inference",
            "fallback_chat_models": [
                "facebook/bart-large-cnn:hf-inference",
                "google/flan-t5-large:hf-inference",
                "HuggingFaceH4/zephyr-7b-beta:hf-inference",
            ],
            "api_url": "https://router.huggingface.co/hf-inference/models/HuggingFaceTB/SmolLM3-3B",
            "fallback_model_ids": [
                "HuggingFaceTB/SmolLM3-3B",
                "facebook/bart-large-cnn",
                "google/flan-t5-large",
            ],
        },
    }

    DEFAULT_CONFIG = {
        "model_profile": "report_generation",
        "apply_profile_defaults": True,
        "use_chat_completions": True,
        "chat_api_url": "https://router.huggingface.co/v1/chat/completions",
        "chat_model": "openbmb/AgentCPM-Report:hf-inference",
        "fallback_chat_models": [
            "HuggingFaceTB/SmolLM3-3B:hf-inference",
            "Qwen/Qwen2.5-7B-Instruct:novita",
            "mistralai/Mistral-7B-Instruct-v0.3:featherless-ai",
        ],
        "api_url": "https://router.huggingface.co/hf-inference/models/openbmb/AgentCPM-Report",
        "fallback_model_ids": [
            "openbmb/AgentCPM-Report",
            "HuggingFaceTB/SmolLM3-3B",
            "Qwen/Qwen2.5-7B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "HuggingFaceH4/zephyr-7b-beta",
        ],
        "try_router_endpoint": True,
        "token_env_var": "HF_API_TOKEN",
        "api_token": "",
        "max_new_tokens": 900,
        "temperature": 0.2,
        "top_p": 0.9,
        "timeout_sec": 120,
        "wait_for_model": True,
    }

    DEFAULT_PROMPT = """Tu es un Data Scientist senior.
Analyse les éléments du diagnostic et fournis un rapport structuré en français.

Contraintes:
- Reste factuel et exploitable pour la décision.
- Mentionne les limites statistiques quand nécessaire.
- Propose des actions concrètes priorisées.
- Format attendu:
1) Résumé exécutif (5-8 lignes)
2) Qualité des données
3) Tendances statistiques majeures
4) Risques et biais potentiels
5) Recommandations opérationnelles (court / moyen terme)
6) Prochaines analyses recommandées
"""

    @staticmethod
    def ensure_default_files(config_path: str, prompt_path: str):
        if not os.path.exists(config_path):
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(HuggingFaceReportAssistant.DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        if not os.path.exists(prompt_path):
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(HuggingFaceReportAssistant.DEFAULT_PROMPT)

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        merged = dict(HuggingFaceReportAssistant.DEFAULT_CONFIG)
        merged.update(cfg or {})
        merged = HuggingFaceReportAssistant._apply_model_profile(merged)
        return merged

    @staticmethod
    def get_profile_names() -> List[str]:
        return list(HuggingFaceReportAssistant.MODEL_PROFILES.keys())

    @staticmethod
    def _apply_model_profile(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Applique les valeurs du profil modèle sélectionné."""
        if not bool(cfg.get("apply_profile_defaults", True)):
            return cfg

        profile_name = str(cfg.get("model_profile", "")).strip()
        profile = HuggingFaceReportAssistant.MODEL_PROFILES.get(profile_name)
        if not profile:
            return cfg

        cfg = dict(cfg)
        cfg["chat_model"] = str(profile.get("chat_model", cfg.get("chat_model", ""))).strip()

        fallback_chat_profile = profile.get("fallback_chat_models", [])
        fallback_chat_cfg = cfg.get("fallback_chat_models", [])
        merged_fallback_chat = []
        if isinstance(fallback_chat_profile, list):
            merged_fallback_chat.extend(str(v).strip() for v in fallback_chat_profile if str(v).strip())
        if isinstance(fallback_chat_cfg, list):
            merged_fallback_chat.extend(str(v).strip() for v in fallback_chat_cfg if str(v).strip())
        cfg["fallback_chat_models"] = HuggingFaceReportAssistant._dedupe_keep_order(merged_fallback_chat)

        api_url = str(profile.get("api_url", "")).strip()
        if api_url:
            cfg["api_url"] = api_url

        fallback_ids_profile = profile.get("fallback_model_ids", [])
        fallback_ids_cfg = cfg.get("fallback_model_ids", [])
        merged_fallback_ids = []
        if isinstance(fallback_ids_profile, list):
            merged_fallback_ids.extend(str(v).strip().strip("/") for v in fallback_ids_profile if str(v).strip())
        if isinstance(fallback_ids_cfg, list):
            merged_fallback_ids.extend(str(v).strip().strip("/") for v in fallback_ids_cfg if str(v).strip())
        cfg["fallback_model_ids"] = HuggingFaceReportAssistant._dedupe_keep_order(merged_fallback_ids)
        return cfg

    @staticmethod
    def load_prompt(prompt_path: str) -> str:
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            return prompt or HuggingFaceReportAssistant.DEFAULT_PROMPT
        except Exception:
            return HuggingFaceReportAssistant.DEFAULT_PROMPT

    @staticmethod
    def read_context_file(context_path: Optional[str], max_chars: int = 12000) -> str:
        if not context_path:
            return ""
        if not os.path.exists(context_path):
            return ""

        ext = os.path.splitext(context_path)[1].lower()
        try:
            if ext in [".txt", ".md", ".log", ".rst"]:
                with open(context_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read(max_chars)
            if ext == ".json":
                with open(context_path, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                return json.dumps(data, ensure_ascii=False, indent=2)[:max_chars]
            if ext == ".csv":
                sample_df = pd.read_csv(context_path, low_memory=False).head(50)
                with pd.option_context("display.max_columns", None, "display.width", 220):
                    return sample_df.to_string()[:max_chars]
        except Exception:
            return ""
        return ""

    @staticmethod
    def _extract_generated_text(response_json: Any) -> str:
        if isinstance(response_json, list) and response_json:
            first = response_json[0]
            if isinstance(first, dict):
                for key in ("generated_text", "summary_text", "text"):
                    if key in first and first[key]:
                        return str(first[key]).strip()
            return str(first).strip()
        if isinstance(response_json, dict):
            if "error" in response_json:
                raise ValueError(f"API Hugging Face: {response_json['error']}")
            for key in ("generated_text", "summary_text", "text", "output_text"):
                if key in response_json and response_json[key]:
                    return str(response_json[key]).strip()
        return str(response_json).strip()

    @staticmethod
    def _extract_chat_content(response_json: Any) -> str:
        """Extrait le texte d'une réponse OpenAI-compatible chat/completions."""
        if isinstance(response_json, dict):
            if "error" in response_json:
                err = response_json["error"]
                if isinstance(err, dict):
                    msg = err.get("message") or str(err)
                else:
                    msg = str(err)
                raise ValueError(f"API Hugging Face chat: {msg}")

            choices = response_json.get("choices", [])
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                content = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            parts.append(str(part["text"]))
                        elif isinstance(part, str):
                            parts.append(part)
                    joined = "\n".join(p for p in parts if p).strip()
                    if joined:
                        return joined
        return ""

    @staticmethod
    def _dedupe_keep_order(items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            key = str(item).strip()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
        return out

    @staticmethod
    def _build_candidate_chat_models(cfg: Dict[str, Any]) -> List[str]:
        models: List[str] = []
        primary = str(cfg.get("chat_model", "")).strip()
        if primary:
            models.append(primary)

        fallback_chat = cfg.get("fallback_chat_models", [])
        if isinstance(fallback_chat, list):
            models.extend(str(m).strip() for m in fallback_chat if str(m).strip())

        # Backward compatibility: transforme fallback_model_ids en ":hf-inference"
        fallback_ids = cfg.get("fallback_model_ids", [])
        if isinstance(fallback_ids, list):
            for mid in fallback_ids:
                model_id = str(mid).strip().strip("/")
                if model_id:
                    models.append(f"{model_id}:hf-inference")

        return HuggingFaceReportAssistant._dedupe_keep_order(models)

    @staticmethod
    def _model_id_from_url(url: str) -> str:
        marker = "/models/"
        if marker not in url:
            return ""
        return url.split(marker, 1)[1].strip().strip("/")

    @staticmethod
    def _build_candidate_urls(cfg: Dict[str, Any]) -> List[str]:
        primary_url = str(cfg.get("api_url", "")).strip()
        try_router = bool(cfg.get("try_router_endpoint", True))
        router_prefix = "https://router.huggingface.co/hf-inference/models/"
        legacy_prefix = "https://api-inference.huggingface.co/models/"

        candidates: List[str] = []
        if primary_url:
            candidates.append(primary_url)

        if try_router and primary_url.startswith(legacy_prefix):
            candidates.append(primary_url.replace(legacy_prefix, router_prefix))
        if try_router and primary_url.startswith(router_prefix):
            candidates.append(primary_url.replace(router_prefix, legacy_prefix))

        fallback_models = cfg.get("fallback_model_ids", [])
        if isinstance(fallback_models, list):
            for model_id in fallback_models:
                mid = str(model_id).strip().strip("/")
                if not mid:
                    continue
                candidates.append(f"{router_prefix}{mid}")
                candidates.append(f"{legacy_prefix}{mid}")

        # Déduplication en conservant l'ordre
        seen = set()
        unique: List[str] = []
        for u in candidates:
            if u and u not in seen:
                seen.add(u)
                unique.append(u)
        return unique

    @staticmethod
    def generate_analysis(
        config_path: str,
        prompt_path: str,
        dataset_summary: str,
        latest_results: str,
        context_file: Optional[str] = None
    ) -> str:
        try:
            import requests
        except ImportError as exc:
            raise ImportError("Installez requests: pip install requests") from exc

        cfg = HuggingFaceReportAssistant.load_config(config_path)
        token_env = str(cfg.get("token_env_var", "HF_API_TOKEN")).strip()
        token = str(cfg.get("api_token", "")).strip()

        # Tolère une mauvaise saisie du type: HF_API_TOKEN="hf_xxx"
        if "=" in token_env:
            left, right = token_env.split("=", 1)
            parsed_env = left.strip() or "HF_API_TOKEN"
            inline_token = right.strip().strip('"').strip("'")
            token_env = parsed_env
            if not token and inline_token.startswith("hf_"):
                token = inline_token

        if not token:
            token = os.environ.get(token_env, "").strip()
        if not token:
            raise ValueError(
                "Token Hugging Face manquant. "
                f"Configurez token_env_var='{token_env}' puis exportez la variable "
                f"(ex: export {token_env}='hf_xxx')."
            )

        system_prompt = HuggingFaceReportAssistant.load_prompt(prompt_path)
        business_context = HuggingFaceReportAssistant.read_context_file(context_file)

        payload_context = {
            "resume_dataset": dataset_summary,
            "resultats_recents": latest_results[:10000],
            "contexte_metier": business_context[:8000] if business_context else "Aucun contexte additionnel fourni",
        }

        final_prompt = textwrap.dedent(
            f"""
            {system_prompt}

            === CONTEXTE EDA ===
            {json.dumps(payload_context, ensure_ascii=False, indent=2)}
            """
        ).strip()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        body_text_generation = {
            "inputs": final_prompt,
            "parameters": {
                "max_new_tokens": int(cfg.get("max_new_tokens", 900)),
                "temperature": float(cfg.get("temperature", 0.2)),
                "top_p": float(cfg.get("top_p", 0.9)),
                "return_full_text": False,
            },
            "options": {"wait_for_model": bool(cfg.get("wait_for_model", True))}
        }

        timeout_sec = int(cfg.get("timeout_sec", 120))
        errors: List[str] = []

        # Mode recommandé Hugging Face (OpenAI-compatible chat completions)
        if bool(cfg.get("use_chat_completions", True)):
            chat_api_url = str(cfg.get("chat_api_url", "https://router.huggingface.co/v1/chat/completions")).strip()
            chat_models = HuggingFaceReportAssistant._build_candidate_chat_models(cfg)

            for model in chat_models:
                try:
                    chat_payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": json.dumps(payload_context, ensure_ascii=False, indent=2)}
                        ],
                        "temperature": float(cfg.get("temperature", 0.2)),
                        "top_p": float(cfg.get("top_p", 0.9)),
                        "max_tokens": int(cfg.get("max_new_tokens", 900)),
                    }

                    response = requests.post(
                        chat_api_url,
                        headers=headers,
                        json=chat_payload,
                        timeout=timeout_sec,
                    )

                    if response.status_code in (404, 410):
                        errors.append(f"{response.status_code} {model}")
                        continue

                    response.raise_for_status()
                    generated = HuggingFaceReportAssistant._extract_chat_content(response.json())
                    if generated:
                        return generated
                    errors.append(f"Réponse vide chat: {model}")
                except requests.HTTPError as e:
                    status = e.response.status_code if e.response is not None else "HTTP"
                    errors.append(f"{status} {model}")
                    continue
                except Exception as e:
                    errors.append(f"{model}: {str(e)}")
                    continue

        # Fallback historique text-generation endpoint
        urls = HuggingFaceReportAssistant._build_candidate_urls(cfg)
        for api_url in urls:
            model_id = HuggingFaceReportAssistant._model_id_from_url(api_url) or api_url
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=body_text_generation,
                    timeout=timeout_sec,
                )

                if response.status_code in (404, 410):
                    errors.append(f"{response.status_code} {model_id}")
                    continue

                response.raise_for_status()
                generated = HuggingFaceReportAssistant._extract_generated_text(response.json())
                if generated:
                    return generated
                errors.append(f"Réponse vide text-gen: {model_id}")
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else "HTTP"
                errors.append(f"{status} {model_id}")
                continue
            except Exception as e:
                errors.append(f"{model_id}: {str(e)}")
                continue

        details = "; ".join(errors[:6]) if errors else "aucun détail"
        raise ValueError(
            "Aucun endpoint Hugging Face disponible pour le rapport IA. "
            f"Tentatives: {details}"
        )
