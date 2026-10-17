'use client'
import { useState, useEffect, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { Plus, Eye, EyeOff, Loader2 } from 'lucide-react'
import { fetchMenu, toggleMenuItemAvailability, createMenuItem, patchMenuItem } from '../../../services/platformApi'

export default function MenuView() {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [nutritionItemId, setNutritionItemId] = useState(null)
  const [nutritionForm, setNutritionForm] = useState({})
  const [form, setForm] = useState({ name: '', description: '', type: 'VEG', price_cents: '', available: true })

  const load = useCallback(async () => {
    try {
      const data = await fetchMenu()
      setItems(data)
    } catch {
      toast.error('Failed to load menu')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  async function handleToggle(item) {
    try {
      await toggleMenuItemAvailability(item.id, !item.available)
      setItems(prev => prev.map(i => i.id === item.id ? { ...i, available: !i.available } : i))
      toast.success(`${item.name} → ${item.available ? 'unavailable' : 'available'}`)
    } catch (err) {
      toast.error(err.message)
    }
  }

  async function handleCreate(e) {
    e.preventDefault()
    try {
      await createMenuItem({
        ...form,
        price_cents: parseInt(form.price_cents) * 100,
      })
      toast.success('Item created')
      setShowForm(false)
      setForm({ name: '', description: '', type: 'VEG', price_cents: '', available: true })
      load()
    } catch (err) {
      toast.error(err.message)
    }
  }

  function editNutrition(item) {
    setNutritionItemId(item.id)
    setNutritionForm({
      energy_kcal: item.energy_kcal ?? '',
      protein_g: item.protein_g ?? '',
      carbohydrates_g: item.carbohydrates_g ?? '',
      fat_g: item.fat_g ?? '',
      fiber_g: item.fiber_g ?? '',
      nutrition_serving_g: item.nutrition_serving_g ?? '',
      nutrition_source: item.nutrition_source || '',
      nutrition_status: item.nutrition_status || 'UNKNOWN',
    })
  }

  async function saveNutrition(event, item) {
    event.preventDefault()
    const payload = { ...nutritionForm }
    for (const key of ['energy_kcal', 'protein_g', 'carbohydrates_g', 'fat_g', 'fiber_g', 'nutrition_serving_g']) {
      payload[key] = payload[key] === '' ? null : Number(payload[key])
    }
    try {
      await patchMenuItem(item.id, payload)
      toast.success('Nutrition data updated')
      setNutritionItemId(null)
      await load()
    } catch (err) {
      toast.error(err.message)
    }
  }

  if (loading) return <div className="page-loading"><Loader2 className="spin" size={24} />Loading menu...</div>

  const byType = { VEG: [], NON_VEG: [], EGG: [], VEGAN: [], '': [] }
  items.forEach(i => { (byType[i.type || ''] ||= []).push(i) })

  return (
    <div className="menu-view">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>{items.length} items</span>
        <button className="btn btn-primary btn-sm" onClick={() => setShowForm(v => !v)}>
          <Plus size={14} /> Add Item
        </button>
      </div>

      {showForm && (
        <form onSubmit={handleCreate} className="menu-form">
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Name</label>
              <input className="form-input" value={form.name} onChange={e => setForm(p => ({ ...p, name: e.target.value }))} required />
            </div>
            <div className="form-group">
              <label className="form-label">Price (₹)</label>
              <input className="form-input" type="number" value={form.price_cents} onChange={e => setForm(p => ({ ...p, price_cents: e.target.value }))} required />
            </div>
            <div className="form-group">
              <label className="form-label">Type</label>
              <select className="form-input" value={form.type} onChange={e => setForm(p => ({ ...p, type: e.target.value }))}>
                {['VEG', 'NON_VEG', 'EGG', 'VEGAN'].map(t => <option key={t}>{t}</option>)}
              </select>
            </div>
          </div>
          <div className="form-group">
            <label className="form-label">Description</label>
            <input className="form-input" value={form.description} onChange={e => setForm(p => ({ ...p, description: e.target.value }))} />
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button type="submit" className="btn btn-primary btn-sm">Save</button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setShowForm(false)}>Cancel</button>
          </div>
        </form>
      )}

      <div className="menu-list">
        {items.map(item => (
          <div key={item.id}>
            <div className="menu-item-row">
              <div className="menu-item-info">
                <span className={`veg-dot ${item.type === 'VEG' || item.type === 'VEGAN' ? 'veg' : 'non-veg'}`} />
                <div>
                  <div style={{ fontWeight: 600 }}>{item.name}</div>
                  {item.description && <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{item.description}</div>}
                </div>
              </div>
              <div className="menu-item-actions">
                <span style={{ fontWeight: 700 }}>₹{((item.price_cents || 0) / 100).toFixed(0)}</span>
                <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 8, background: 'var(--bg-overlay)' }}>{item.type}</span>
                <button
                  className={`btn btn-sm ${item.available ? 'btn-ghost' : 'btn-secondary'}`}
                  onClick={() => handleToggle(item)}
                  title={item.available ? 'Mark unavailable' : 'Mark available'}
                >
                  {item.available ? <Eye size={14} /> : <EyeOff size={14} />}
                  {item.available ? 'Available' : 'Unavailable'}
                </button>
                <button className="btn btn-sm btn-ghost" onClick={() => editNutrition(item)}>
                  Nutrition: {item.nutrition_status || 'UNKNOWN'}
                </button>
              </div>
            </div>
            {nutritionItemId === item.id && <form className="menu-form" onSubmit={event => saveNutrition(event, item)}>
              <strong>Nutrition per serving · {item.name}</strong>
              <p>Use a documented restaurant or supplier source. Only mark values verified after checking that evidence.</p>
              <div className="form-row">
                {[
                  ['energy_kcal', 'Energy (kcal)'],
                  ['protein_g', 'Protein (g)'],
                  ['carbohydrates_g', 'Carbohydrates (g)'],
                  ['fat_g', 'Fat (g)'],
                  ['fiber_g', 'Fibre (g)'],
                  ['nutrition_serving_g', 'Serving size (g)'],
                ].map(([key, label]) => <div className="form-group" key={key}>
                  <label className="form-label" htmlFor={`nutrition-${item.id}-${key}`}>{label}</label>
                  <input
                    id={`nutrition-${item.id}-${key}`}
                    className="form-input"
                    type="number"
                    min="0"
                    step="0.1"
                    value={nutritionForm[key]}
                    onChange={event => setNutritionForm({ ...nutritionForm, [key]: event.target.value })}
                  />
                </div>)}
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label" htmlFor={`nutrition-${item.id}-status`}>Data status</label>
                  <select id={`nutrition-${item.id}-status`} className="form-input" value={nutritionForm.nutrition_status} onChange={event => setNutritionForm({ ...nutritionForm, nutrition_status: event.target.value })}>
                    <option value="UNKNOWN">Unknown</option><option value="ESTIMATED">Estimated</option><option value="VERIFIED">Verified</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label" htmlFor={`nutrition-${item.id}-source`}>Source / evidence</label>
                  <input id={`nutrition-${item.id}-source`} className="form-input" maxLength={500} value={nutritionForm.nutrition_source} onChange={event => setNutritionForm({ ...nutritionForm, nutrition_source: event.target.value })} />
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button type="submit" className="btn btn-primary btn-sm">Save nutrition</button>
                <button type="button" className="btn btn-ghost btn-sm" onClick={() => setNutritionItemId(null)}>Cancel</button>
              </div>
            </form>}
          </div>
        ))}
      </div>
    </div>
  )
}
