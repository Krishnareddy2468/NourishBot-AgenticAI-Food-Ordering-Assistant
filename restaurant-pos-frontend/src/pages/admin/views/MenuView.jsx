import { useState, useEffect, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { Plus, Eye, EyeOff, Loader2 } from 'lucide-react'
import { fetchMenu, toggleMenuItemAvailability, createMenuItem } from '../../../services/platformApi'

export default function MenuView() {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState({ name: '', description: '', type: 'VEG', price_cents: '', available: true })

  const load = useCallback(async () => {
    try {
      const data = await fetchMenu()
      setItems(data)
    } catch (err) {
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
          <div key={item.id} className="menu-item-row">
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
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
