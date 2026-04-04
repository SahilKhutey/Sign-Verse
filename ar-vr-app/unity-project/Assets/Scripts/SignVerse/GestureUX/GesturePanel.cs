using UnityEngine;
using TMPro;
using System;
using System.Collections.Generic;

namespace SignVerse.GestureUX
{
    /// <summary>
    /// GesturePanel — A floating AR panel with gesture-selectable items.
    ///
    /// Designed for minimal clutter:
    ///   - Max 5 items per panel (cognitive limit)
    ///   - Active item highlighted with a subtle glow ring
    ///   - No scrollbars — gesture up/down navigates items
    ///   - Auto-dismisses if user looks away for > 5 seconds
    /// </summary>
    public class GesturePanel : MonoBehaviour
    {
        [Header("Content")]
        public string panelTitle = "Menu";
        public List<PanelItem> items = new List<PanelItem>();

        [Header("Visual")]
        public TextMeshPro titleText;
        public Transform   itemContainer;
        public GameObject  selectionIndicator;  // Highlight ring
        public float       itemSpacing = 0.07f;
        public Color       activeColor   = new Color(0.2f, 0.8f, 1f, 1f);
        public Color       inactiveColor = new Color(0.9f, 0.9f, 0.9f, 0.5f);

        [Header("Auto-Dismiss")]
        public Transform  headTransform;
        public float      autoDismissAngle   = 65f;  // Degrees off-axis
        public float      autoDismissTimeout = 5f;

        // ── Events ─────────────────────────────────────────────────────────────
        public event Action<PanelItem> OnItemSelected;

        // ── State ──────────────────────────────────────────────────────────────
        private int   _selectedIndex = 0;
        private float _offAxisTimer  = 0f;
        private List<TextMeshPro> _itemLabels = new List<TextMeshPro>();

        // ── Lifecycle ──────────────────────────────────────────────────────────
        void OnEnable()
        {
            _selectedIndex = 0;
            RebuildItemList();
            RefreshHighlight();
        }

        void Update()
        {
            CheckAutoDismiss();
        }

        // ── Public API ─────────────────────────────────────────────────────────

        public void ScrollUp()
        {
            _selectedIndex = Mathf.Clamp(_selectedIndex - 1, 0, items.Count - 1);
            RefreshHighlight();
        }

        public void ScrollDown()
        {
            _selectedIndex = Mathf.Clamp(_selectedIndex + 1, 0, items.Count - 1);
            RefreshHighlight();
        }

        public void ConfirmSelection()
        {
            if (_selectedIndex >= 0 && _selectedIndex < items.Count)
            {
                var item = items[_selectedIndex];
                item.onSelected?.Invoke();
                OnItemSelected?.Invoke(item);
                Debug.Log($"[GesturePanel] Selected: {item.label}");
            }
        }

        // ── Internal ───────────────────────────────────────────────────────────

        private void RebuildItemList()
        {
            if (titleText != null) titleText.text = panelTitle;

            // Clear old labels
            foreach (var lbl in _itemLabels)
                if (lbl != null) Destroy(lbl.gameObject);
            _itemLabels.Clear();

            if (itemContainer == null) return;

            // Spawn item labels
            for (int i = 0; i < Mathf.Min(items.Count, 5); i++)
            {
                var go  = new GameObject($"Item_{i}");
                go.transform.SetParent(itemContainer, false);
                go.transform.localPosition = new Vector3(0, -i * itemSpacing, 0);

                var tmp = go.AddComponent<TextMeshPro>();
                tmp.text      = items[i].label;
                tmp.fontSize  = 0.045f;
                tmp.alignment = TextAlignmentOptions.Left;
                tmp.color     = inactiveColor;
                _itemLabels.Add(tmp);
            }
        }

        private void RefreshHighlight()
        {
            for (int i = 0; i < _itemLabels.Count; i++)
            {
                if (_itemLabels[i] == null) continue;
                _itemLabels[i].color     = (i == _selectedIndex) ? activeColor : inactiveColor;
                _itemLabels[i].fontStyle = (i == _selectedIndex) ? FontStyles.Bold : FontStyles.Normal;
            }

            // Move selection indicator
            if (selectionIndicator != null && _selectedIndex < _itemLabels.Count && _itemLabels[_selectedIndex] != null)
            {
                selectionIndicator.transform.localPosition = new Vector3(-0.05f, -_selectedIndex * itemSpacing, 0);
                selectionIndicator.SetActive(true);
            }
        }

        private void CheckAutoDismiss()
        {
            if (headTransform == null) return;

            Vector3 toPanel  = (transform.position - headTransform.position).normalized;
            float   angle    = Vector3.Angle(headTransform.forward, toPanel);

            if (angle > autoDismissAngle)
            {
                _offAxisTimer += Time.deltaTime;
                if (_offAxisTimer >= autoDismissTimeout)
                    gameObject.SetActive(false);
            }
            else
            {
                _offAxisTimer = 0f;
            }
        }
    }

    [Serializable]
    public class PanelItem
    {
        public string              label;
        public UnityEngine.Events.UnityEvent onSelected;
    }
}
